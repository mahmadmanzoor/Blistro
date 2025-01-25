import { handleAuth } from "@kinde-oss/kinde-auth-nextjs/server";
import { NextRequest } from "next/server";

export async function GET(
  request: NextRequest,
  // Declare that `params` itself is a promise returning an object that might have 'kindeAuth'
  { params }: { params: Promise<{ kindeAuth?: string }> }
) {
  // Await the entire params object
  const { kindeAuth } = await params;

  if (!kindeAuth) {
    return new Response("Endpoint not found", { status: 404 });
  }

  return handleAuth(request, kindeAuth);
}
