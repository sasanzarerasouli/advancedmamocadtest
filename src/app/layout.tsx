import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "MammoCAD Lab – mammography analysis & algorithm reconstruction",
  description: "Executable mammography CAD research pipeline: anatomy exclusion, mass segmentation, calcification detection, Python design and validation plan.",
};

const NAV = [
  { href: "/", label: "Analyzer (live)" },
  { href: "/report", label: "Image report" },
  { href: "/methodology", label: "Methodology" },
  { href: "/python", label: "Python modules" },
  { href: "/integration", label: "MammographyCAD integration" },
  { href: "/validation", label: "Validation" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-100 text-slate-900 antialiased">
        <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950 text-slate-100">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-1 px-4 py-3">
            <span className="font-semibold tracking-tight">
              MammoCAD <span className="text-teal-400">Lab</span>
            </span>
            <nav className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-slate-300">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className="hover:text-white">
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
        <footer className="mx-auto max-w-7xl px-4 pb-10 text-xs text-slate-500">
          Research and CAD-development material. Not a medical device; not for clinical decision-making; does not replace radiologist interpretation.
        </footer>
      </body>
    </html>
  );
}
