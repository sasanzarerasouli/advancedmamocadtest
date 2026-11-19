import { NextResponse } from "next/server";
import { desc } from "drizzle-orm";
import { db } from "@/db";
import { analyses } from "@/db/schema";

export const dynamic = "force-dynamic";

export async function GET() {
  const rows = await db.select().from(analyses).orderBy(desc(analyses.createdAt)).limit(25);
  return NextResponse.json({ rows });
}

interface Body {
  filename?: string;
  width?: number;
  height?: number;
  projection?: string;
  pectoralFound?: boolean;
  massCount?: number;
  calcCandidateCount?: number;
  calcClusterCount?: number;
  runtimeMs?: number;
  params?: unknown;
  findings?: unknown;
}

export async function POST(req: Request) {
  let b: Body;
  try {
    b = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (!b.filename || !b.width || !b.height) return NextResponse.json({ error: "filename, width, height required" }, { status: 400 });
  const [row] = await db
    .insert(analyses)
    .values({
      filename: String(b.filename).slice(0, 200),
      width: Math.round(b.width),
      height: Math.round(b.height),
      projection: String(b.projection ?? "unknown").slice(0, 200),
      pectoralFound: b.pectoralFound ? 1 : 0,
      massCount: b.massCount ?? 0,
      calcCandidateCount: b.calcCandidateCount ?? 0,
      calcClusterCount: b.calcClusterCount ?? 0,
      runtimeMs: b.runtimeMs ?? 0,
      params: b.params ?? {},
      findings: b.findings ?? {},
    })
    .returning();
  return NextResponse.json({ row });
}

export async function DELETE() {
  await db.delete(analyses);
  return NextResponse.json({ ok: true });
}
