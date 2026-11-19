import type { ReactNode } from "react";

export function Page({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <article className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {intro && <div className="mt-2 max-w-4xl text-sm leading-relaxed text-slate-600">{intro}</div>}
      </header>
      {children}
    </article>
  );
}

export function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="space-y-3 rounded-xl bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="space-y-3 text-sm leading-relaxed text-slate-700">{children}</div>
    </section>
  );
}

export function Callout({ kind = "info", children }: { kind?: "info" | "warn" | "ok"; children: ReactNode }) {
  const c = kind === "warn" ? "border-amber-300 bg-amber-50 text-amber-900" : kind === "ok" ? "border-emerald-300 bg-emerald-50 text-emerald-900" : "border-sky-300 bg-sky-50 text-sky-900";
  return <div className={`rounded-lg border p-3 text-sm ${c}`}>{children}</div>;
}

export function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="bg-slate-50 text-slate-500">
          <tr>
            {head.map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-100 align-top">
              {r.map((c, j) => (
                <td key={j} className="px-2 py-1.5">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Code({ children }: { children: string }) {
  return <pre className="overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-snug text-slate-100">{children}</pre>;
}

export function M({ children }: { children: ReactNode }) {
  return <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[12px] text-slate-800">{children}</code>;
}
