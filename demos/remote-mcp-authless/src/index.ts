import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";
import { z } from "zod";

interface ImageWorkerEnv {
  OPENAI_API_KEY: string;
  URL_SECRET: string;
  IMAGES: R2Bucket;
}

interface ImageResponse {
  data?: Array<{ b64_json?: string }>;
  error?: { code?: string; param?: string; message?: string };
}

interface ChatFile {
  download_url: string;
  file_id: string;
  mime_type?: string;
  file_name?: string;
}

const MODELS = [
  "gpt-image-2.6-sunburst",
  "gpt-image-2.6-flare",
  "gpt-image-2.5-flare",
  "gpt-image-2",
  "gpt-image-1.5",
  "gpt-image-1",
  "gpt-image-1-mini",
] as const;

const DEFAULT_MODEL = MODELS[0];

const RATIOS = ["1:1", "16:9", "4:3", "3:4", "9:16"] as const;
type Ratio = (typeof RATIOS)[number];

const FIXED_SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"];

const STANDARD_SIZE: Record<Ratio, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x1024",
  "4:3": "1536x1024",
  "3:4": "1024x1536",
  "9:16": "1024x1536",
};

const EXACT_SIZE: Record<Ratio, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x864",
  "4:3": "1536x1152",
  "3:4": "1152x1536",
  "9:16": "864x1536",
};

const MAX_REFERENCES = 16;
const MAX_REFERENCE_BYTES = 20 * 1024 * 1024; // 20 МБ на один файл

function supportsCustomSizes(model: string): boolean {
  return model.startsWith("gpt-image-2");
}

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// Скачивает вложение из чата и превращает в data URL
async function chatFileToDataUrl(file: ChatFile): Promise<string> {
  const name = file.file_name ?? file.file_id;

  if (!file.download_url.startsWith("https://")) {
    throw new Error(`Файл ${name}: ссылка для скачивания должна быть https://`);
  }

  const res = await fetch(file.download_url);
  if (!res.ok) {
    throw new Error(`Не удалось скачать файл ${name}: HTTP ${res.status}`);
  }

  const bytes = new Uint8Array(await res.arrayBuffer());
  if (bytes.length > MAX_REFERENCE_BYTES) {
    throw new Error(`Файл ${name} больше 20 МБ.`);
  }

  const mime =
    file.mime_type ?? res.headers.get("content-type")?.split(";")[0]?.trim() ?? "image/png";
  if (!mime.startsWith("image/")) {
    throw new Error(`Файл ${name} не является изображением (${mime}).`);
  }

  return `data:${mime};base64,${toBase64(bytes)}`;
}

const chatFileSchema = z
  .object({
    download_url: z.string(),
    file_id: z.string(),
    mime_type: z.string().optional(),
    file_name: z.string().optional(),
  })
  .strict();

function createImageServer(env: ImageWorkerEnv, origin: string) {
  const server = new McpServer({
    name: "openai-image-r2",
    version: "1.4.0",
  });

  server.registerTool(
    "generate_image",
    {
      description:
        "Создаёт изображение через OpenAI Images API и возвращает ссылку на PNG. " +
        "Размер: если указан size (например 1536x1024) — используется он; " +
        "иначе, если указан aspect_ratio (1:1, 16:9, 4:3, 3:4, 9:16) — размер подбирается по нему; " +
        "если не указано ничего — размер выбирает модель (auto). " +
        "Референсы: reference_files — изображения, прикреплённые пользователем в чат; " +
        "reference_images — публичные ссылки https:// или data:image/... . " +
        "С референсами изображение создаётся на их основе.",
      inputSchema: z.object({
        prompt: z.string().trim().min(1),
        model: z.enum(MODELS).default(DEFAULT_MODEL),
        size: z
          .string()
          .regex(/^(auto|\d{3,4}x\d{3,4})$/, "Формат: ШИРИНАxВЫСОТА, например 1536x1024, или auto")
          .optional()
          .describe("Точный размер ШИРИНАxВЫСОТА. Имеет приоритет над aspect_ratio."),
        aspect_ratio: z
          .enum(RATIOS)
          .optional()
          .describe("Соотношение сторон. Используется, только если size не указан."),
        quality: z.enum(["auto", "low", "medium", "high"]).default("auto"),
        reference_files: z
          .array(chatFileSchema)
          .max(MAX_REFERENCES)
          .optional()
          .describe("Изображения-референсы, прикреплённые пользователем в чат."),
        reference_images: z
          .array(z.string().regex(/^(https:\/\/|data:image\/)/, "Нужна ссылка https:// или data:image/..."))
          .max(MAX_REFERENCES)
          .optional()
          .describe("Референсы по публичным ссылкам (https:// или data URL)."),
      }),
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
      // Говорим ChatGPT, что это поле принимает файлы из чата
      _meta: {
        "openai/fileParams": ["reference_files"],
      },
    },
    async ({ prompt, model, size, aspect_ratio, quality, reference_files, reference_images }) => {
      // Собираем все референсы в один список
      const references: string[] = [...(reference_images ?? [])];

      try {
        for (const file of reference_files ?? []) {
          references.push(await chatFileToDataUrl(file));
        }
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: error instanceof Error ? error.message : "Ошибка чтения референса." }],
        };
      }

      if (references.length > MAX_REFERENCES) {
        return {
          isError: true,
          content: [{ type: "text", text: `Слишком много референсов: максимум ${MAX_REFERENCES}.` }],
        };
      }

      const hasRefs = references.length > 0;
      const endpoint = hasRefs ? "images/edits" : "images/generations";

      // Размер: size -> aspect_ratio -> auto
      let firstSize: string;
      let fallbackSize: string | null = null;
      let source: "size" | "aspect_ratio" | "auto";

      if (size) {
        source = "size";
        firstSize = size;
        if (!supportsCustomSizes(model) && !FIXED_SIZES.includes(size)) {
          return {
            isError: true,
            content: [
              {
                type: "text",
                text: `Модель ${model} поддерживает только размеры: ${FIXED_SIZES.join(", ")}. Выберите один из них или модель gpt-image-2 / 2.5 / 2.6.`,
              },
            ],
          };
        }
      } else if (aspect_ratio) {
        source = "aspect_ratio";
        firstSize = supportsCustomSizes(model) ? EXACT_SIZE[aspect_ratio] : STANDARD_SIZE[aspect_ratio];
        if (firstSize !== STANDARD_SIZE[aspect_ratio]) fallbackSize = STANDARD_SIZE[aspect_ratio];
      } else {
        source = "auto";
        firstSize = "auto";
      }

      const callApi = async (sizeToUse: string) => {
        const payload: Record<string, unknown> = {
          model,
          prompt,
          size: sizeToUse,
          quality,
          n: 1,
          output_format: "png",
        };
        if (hasRefs) {
          payload.images = references.map((image_url) => ({ image_url }));
        }

        const response = await fetch(`https://api.openai.com/v1/${endpoint}`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.OPENAI_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        const result = await response.json<ImageResponse>().catch(() => null);
        return { response, result };
      };

      let usedSize = firstSize;
      let { response, result } = await callApi(usedSize);

      if (!response.ok && response.status === 400 && fallbackSize) {
        usedSize = fallbackSize;
        ({ response, result } = await callApi(usedSize));
      }

      if (!response.ok) {
        const details = [result?.error?.code, result?.error?.param, result?.error?.message]
          .filter(Boolean)
          .join(" | ");

        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `OpenAI ${endpoint}: HTTP ${response.status}${details ? ` (${details})` : ""}. Модель: ${model}, размер: ${usedSize}, референсов: ${references.length}.`,
            },
          ],
        };
      }

      const image = result?.data?.[0]?.b64_json;
      if (!image) {
        return {
          isError: true,
          content: [{ type: "text", text: "OpenAI не вернул изображение." }],
        };
      }

      const key = `${crypto.randomUUID()}.png`;
      const bytes = Uint8Array.from(atob(image), (character) => character.charCodeAt(0));

      await env.IMAGES.put(key, bytes, {
        httpMetadata: { contentType: "image/png" },
      });

      const how =
        source === "size"
          ? "задан явно"
          : source === "aspect_ratio"
            ? `по соотношению ${aspect_ratio}${fallbackSize && usedSize === fallbackSize ? ", ближайший доступный" : ""}`
            : "выбран автоматически";

      return {
        content: [
          {
            type: "text",
            text: `${origin}/img/${key}\nМодель: ${model}, размер: ${usedSize} (${how}), референсов: ${references.length}`,
          },
        ],
      };
    },
  );

  return server;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const imageMatch = /^\/img\/([0-9a-f-]{36}\.png)$/.exec(url.pathname);

    if (request.method === "GET" && imageMatch) {
      const image = await env.IMAGES.get(imageMatch[1]);
      if (!image) return new Response("Not found", { status: 404 });

      return new Response(image.body, {
        headers: {
          "Content-Type": "image/png",
          "Cache-Control": "public, max-age=86400",
        },
      });
    }

    if (!env.URL_SECRET || url.pathname !== `/${env.URL_SECRET}/mcp`) {
      return new Response("Not found", { status: 404 });
    }

    const origin = url.origin;
    url.pathname = "/mcp";
    const handler = createMcpHandler(() => createImageServer(env, origin));

    return handler(new Request(url.toString(), request), env, ctx);
  },
} satisfies ExportedHandler<ImageWorkerEnv>;
