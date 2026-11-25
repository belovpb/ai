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
  error?: { code?: string };
}

function createImageServer(env: ImageWorkerEnv, origin: string) {
  const server = new McpServer({
    name: "openai-image-r2",
    version: "1.0.0",
  });

  server.registerTool(
    "generate_image",
    {
      description: "Создаёт изображение через OpenAI Images API и возвращает ссылку на PNG.",
      inputSchema: z.object({
        prompt: z.string().trim().min(1),
        size: z.enum(["1024x1024", "1536x1024", "1024x1536"]).default("1024x1024"),
      }),
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async ({ prompt, size }) => {
      const response = await fetch("https://api.openai.com/v1/images/generations", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-image-1",
          prompt,
          size,
          n: 1,
          output_format: "png",
        }),
      });

      const result = await response.json<ImageResponse>().catch(() => null);

      if (!response.ok) {
        const code = result?.error?.code;
        return {
          isError: true,
          content: [{
            type: "text",
            text: `OpenAI Images API: HTTP ${response.status}${code ? ` (${code})` : ""}.`,
          }],
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

      return {
        content: [{ type: "text", text: `${origin}/img/${key}` }],
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
