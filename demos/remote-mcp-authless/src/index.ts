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

// Размеры, которые поддерживают все модели
const FIXED_SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"];

const STANDARD_SIZE: Record<Ratio, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x1024",
  "4:3": "1536x1024",
  "3:4": "1024x1536",
  "9:16": "1024x1536",
};

// Точные размеры для моделей с произвольными разрешениями
const EXACT_SIZE: Record<Ratio, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x864",
  "4:3": "1536x1152",
  "3:4": "1152x1536",
  "9:16": "864x1536",
};

function supportsCustomSizes(model: string): boolean {
  return model.startsWith("gpt-image-2");
}

function createImageServer(env: ImageWorkerEnv, origin: string) {
  const server = new McpServer({
    name: "openai-image-r2",
    version: "1.3.0",
  });

  server.registerTool(
    "generate_image",
    {
      description:
        "Создаёт изображение через OpenAI Images API и возвращает ссылку на PNG. " +
        "Размер: если указан size (например 1536x1024) — используется он; " +
        "иначе, если указан aspect_ratio (1:1, 16:9, 4:3, 3:4, 9:16) — размер подбирается по нему; " +
        "если не указано ничего — размер выбирает модель (auto). " +
        "Можно выбрать модель и передать референсы (ссылки https:// или data:image/...).",
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
        reference_images: z
          .array(z.string().regex(/^(https:\/\/|data:image\/)/, "Нужна ссылка https:// или data:image/..."))
          .max(16)
          .optional()
          .describe("Референсы: до 16 ссылок на изображения (https:// или data URL)"),
      }),
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async ({ prompt, model, size, aspect_ratio, quality, reference_images }) => {
      const hasRefs = !!reference_images?.length;
      const endpoint = hasRefs ? "images/edits" : "images/generations";

      // Определяем размер по схеме: size -> aspect_ratio -> auto
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
          payload.images = reference_images!.map((image_url) => ({ image_url }));
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

      // Запасной вариант только для размеров, подобранных по aspect_ratio
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
              text: `OpenAI ${endpoint}: HTTP ${response.status}${details ? ` (${details})` : ""}. Модель: ${model}, размер: ${usedSize}.`,
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
        content: [{ type: "text", text: `${origin}/img/${key}\nМодель: ${model}, размер: ${usedSize} (${how})` }],
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
