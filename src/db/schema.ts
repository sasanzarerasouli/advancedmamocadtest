import { integer, jsonb, pgTable, real, serial, text, timestamp } from "drizzle-orm/pg-core";

/**
 * Stores only derived, non-pixel summaries of CAD runs (no image data is persisted).
 */
export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  filename: text("filename").notNull(),
  width: integer("width").notNull(),
  height: integer("height").notNull(),
  projection: text("projection").notNull(),
  pectoralFound: integer("pectoral_found").notNull().default(0),
  massCount: integer("mass_count").notNull().default(0),
  calcCandidateCount: integer("calc_candidate_count").notNull().default(0),
  calcClusterCount: integer("calc_cluster_count").notNull().default(0),
  runtimeMs: real("runtime_ms").notNull().default(0),
  params: jsonb("params").notNull(),
  findings: jsonb("findings").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type AnalysisRow = typeof analyses.$inferSelect;
