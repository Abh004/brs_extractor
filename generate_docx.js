/**
 * generate_docx.js
 *
 * Reads output/extracted_requirements.json and output/project_config.json
 * (LLM-generated) to produce a fully structured BRS Word document.
 *
 * All document content — project name, scope, categories, NFRs, constraints,
 * assumptions, definitions — comes from project_config.json. Nothing is
 * hardcoded. The same script works for any software product.
 *
 * Each requirement bullet is now an object: { text, priority } where
 * priority is one of "M" / "S" / "C" (MoSCoW), assigned upstream in the
 * Python pipeline (priority.py) rather than hardcoded here.
 *
 * Usage:
 * node generate_docx.js [requirements.json] [output.docx] [project_config.json]
 * Defaults:
 * output/extracted_requirements.json
 * output/BRS.docx
 * output/project_config.json
 */
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  Header,
  Footer,
  AlignmentType,
  LevelFormat,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,
  VerticalAlign,
  PageNumber,
  PageBreak,
  TabStopType,
  TabStopPosition,
  TableOfContents,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ── CLI args ─────────────────────────────────────────────────────────────────
const INPUT_FILE = process.argv[2] || "output/extracted_requirements.json";
const OUTPUT_FILE = process.argv[3] || "output/BRS.docx";
const CONFIG_FILE = process.argv[4] || "output/project_config.json";

// ── Load files ────────────────────────────────────────────────────────────────
const raw = JSON.parse(fs.readFileSync(INPUT_FILE, "utf8"));
const cfg = JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8"));
const summary = raw.summary;

// Flatten all bullets. Each bullet is { text, priority } — but tolerate
// plain-string bullets too (e.g. an older JSON file generated before the
// priority field existed), so this script doesn't hard-crash on stale input.
const allBullets = raw.requirements
  .flatMap((b) => b.bullets || [])
  .map((b) => (typeof b === "string" ? { text: b, priority: "M" } : b));

// ── Categorization ────────────────────────────────────────────────────────────
// Uses cfg.categories from project_config.json — no hardcoded keywords.
// Each bullet is assigned to the first category whose keywords match.
// If cfg.categories is empty or missing, all bullets go to "General Requirements".
const CATEGORIES =
  cfg.categories && cfg.categories.length > 0
    ? cfg.categories
    : [{ id: "FR-01", title: "General Requirements", keywords: [] }];

function categorize(bullet) {
  const lower = bullet.text.toLowerCase();
  for (const cat of CATEGORIES) {
    if (
      (cat.keywords || []).some((kw) => lower.includes(kw.toLowerCase()))
    ) {
      return cat;
    }
  }
  // Fallback: last category acts as catch-all
  return CATEGORIES[CATEGORIES.length - 1];
}

const categoryMap = {};
for (const cat of CATEGORIES) {
  categoryMap[cat.id] = { ...cat, bullets: [] };
}
for (const bullet of allBullets) {
  const cat = categorize(bullet);
  categoryMap[cat.id].bullets.push(bullet);
}

const populatedCategories = Object.values(categoryMap).filter(
  (c) => c.bullets.length > 0,
);

// ── Style constants ───────────────────────────────────────────────────────────
const BLUE = "1F4E79";
const LIGHT_BLUE = "D6E4F0";
const MID_BLUE = "2E75B6";
const GRAY = "595959";

const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = {
  top: BORDER,
  bottom: BORDER,
  left: BORDER,
  right: BORDER,
};

// ── Element helpers ───────────────────────────────────────────────────────────
const run = (text, opts = {}) =>
  new TextRun({ text: String(text), font: "Arial", ...opts });

const p = (children, opts = {}) => new Paragraph({ children, ...opts });

const heading1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", bold: true })],
  });

const heading2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", bold: true })],
  });

const bulletPara = (text) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [
      new TextRun({ text: String(text), font: "Arial", size: 22 }),
    ],
  });

const spacer = (n = 1) =>
  Array.from(
    { length: n },
    () => new Paragraph({ children: [new TextRun("")] }),
  );

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const cell = (children, width, opts = {}) =>
  new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    ...opts,
    children: Array.isArray(children) ? children : [children],
  });

const headerCell = (text, width) =>
  cell(
    [p([run(text, { bold: true, color: "FFFFFF", size: 20 })])],
    width,
    { shading: { fill: BLUE, type: ShadingType.CLEAR } },
  );

const labelCell = (text, width) =>
  cell([p([run(text, { bold: true, size: 20 })])], width, {
    shading: { fill: LIGHT_BLUE, type: ShadingType.CLEAR },
  });

const valueCell = (text, width) =>
  cell([p([run(text, { size: 20 })])], width);

// Priority cell color: Must = blue (existing accent), Should = gray,
// Could = lighter gray, so the table reads as a real priority gradient
// instead of every row looking identical.
const PRIORITY_COLOR = {
  M: MID_BLUE,
  S: GRAY,
  C: "9B9B9B",
};

const today = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

// ── Header / Footer ───────────────────────────────────────────────────────────
const docHeader = {
  default: new Header({
    children: [
      new Paragraph({
        border: {
          bottom: {
            style: BorderStyle.SINGLE,
            size: 6,
            color: MID_BLUE,
            space: 1,
          },
        },
        children: [
          run("Business Requirements Specification", {
            bold: true,
            color: BLUE,
            size: 20,
          }),
          run(`   |   ${cfg.confidentiality || "CONFIDENTIAL"}`, {
            color: GRAY,
            size: 18,
          }),
        ],
      }),
    ],
  }),
};

const docFooter = {
  default: new Footer({
    children: [
      new Paragraph({
        border: {
          top: {
            style: BorderStyle.SINGLE,
            size: 6,
            color: MID_BLUE,
            space: 1,
          },
        },
        tabStops: [
          { type: TabStopType.RIGHT, position: TabStopPosition.MAX },
        ],
        children: [
          run(today, { color: GRAY, size: 18 }),
          new TextRun({ text: "\t", font: "Arial" }),
          run("Page ", { color: GRAY, size: 18 }),
          new TextRun({
            font: "Arial",
            size: 18,
            children: [PageNumber.CURRENT],
          }),
        ],
      }),
    ],
  }),
};

// ── Cover page ────────────────────────────────────────────────────────────────
const coverPage = [
  ...spacer(6),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      run("BUSINESS REQUIREMENTS SPECIFICATION", {
        bold: true,
        size: 56,
        color: BLUE,
      }),
    ],
  }),
  ...spacer(1),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: {
      bottom: {
        style: BorderStyle.SINGLE,
        size: 12,
        color: MID_BLUE,
        space: 4,
      },
    },
    children: [
      run(cfg.project_name || "Software System", {
        size: 32,
        color: GRAY,
      }),
    ],
  }),
  ...spacer(3),
  new Table({
    width: { size: 6000, type: WidthType.DXA },
    columnWidths: [2400, 3600],
    rows: [
      new TableRow({
        children: [labelCell("Date", 2400), valueCell(today, 3600)],
      }),
      new TableRow({
        children: [
          labelCell("Version", 2400),
          valueCell(cfg.version || "1.0", 3600),
        ],
      }),
      new TableRow({
        children: [
          labelCell("Status", 2400),
          valueCell(cfg.status || "Draft", 3600),
        ],
      }),
      new TableRow({
        children: [
          labelCell("Prepared by", 2400),
          valueCell(cfg.prepared_by || "VLM Pipeline", 3600),
        ],
      }),
      new TableRow({
        children: [
          labelCell("Source", 2400),
          valueCell(
            `Video analysis (${summary.total_chunks} chunks)`,
            3600,
          ),
        ],
      }),
    ],
  }),
  pageBreak(),
];

// ── Section 1: Introduction ───────────────────────────────────────────────────
const defsRows = [
  new TableRow({
    tableHeader: true,
    children: [headerCell("Term", 2200), headerCell("Definition", 7160)],
  }),
  ...(cfg.definitions || []).map(
    ([term, def]) =>
      new TableRow({
        children: [
          cell([p([run(term, { bold: true, size: 20 })])], 2200),
          cell([p([run(def, { size: 20 })])], 7160),
        ],
      }),
  ),
];

const section1 = [
  heading1("1. Introduction"),
  heading2("1.1 Purpose"),
  p([run(cfg.purpose || "", { size: 22 })]),
  ...spacer(1),
  heading2("1.2 Scope"),
  p([run(cfg.scope || "", { size: 22 })]),
  ...spacer(1),
  heading2("1.3 Definitions & Acronyms"),
  ...(cfg.definitions && cfg.definitions.length > 0
    ? [
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2200, 7160],
          rows: defsRows,
        }),
      ]
    : [p([run("See project glossary.", { size: 22 })])]),
  pageBreak(),
];

// ── Section 2: Overall Description ───────────────────────────────────────────
const section2 = [
  heading1("2. Overall Description"),
  heading2("2.1 Product Perspective"),
  p([run(cfg.scope || "", { size: 22 })]),
  ...spacer(1),
  heading2("2.2 User Classes"),
  ...(
    cfg.user_classes || [
      "End Users: interact with the system to complete primary tasks",
    ]
  ).map((uc) => bulletPara(uc)),
  ...spacer(1),
  heading2("2.3 Operating Environment"),
  ...(
    cfg.operating_environment || ["Desktop or web-based application"]
  ).map((env) => bulletPara(env)),
  pageBreak(),
];

// ── Section 3: Functional Requirements ───────────────────────────────────────
const reqIdCounter = { n: 1 };
const nextId = () => `REQ-${String(reqIdCounter.n++).padStart(3, "0")}`;

const section3Children = [
  heading1("3. Functional Requirements"),
  p([
    run(
      "The following functional requirements are organized by feature area. " +
        "Each requirement was extracted from video content and refined through " +
        "automated multi-judge analysis. Priority follows MoSCoW (Must / Should / " +
        "Could Have), classified by business criticality rather than literal wording.",
      { size: 22 },
    ),
  ]),
  ...spacer(1),
];

for (const cat of populatedCategories) {
  section3Children.push(heading2(`${cat.id}: ${cat.title}`));
  const rows = [
    new TableRow({
      tableHeader: true,
      children: [
        headerCell("ID", 1200),
        headerCell("Requirement", 7560),
        headerCell("P", 600),
      ],
    }),
    ...cat.bullets.map((bullet, idx) => {
      const shade = {
        fill: idx % 2 === 0 ? "FFFFFF" : "F5F9FC",
        type: ShadingType.CLEAR,
      };
      const priority = bullet.priority || "M";
      return new TableRow({
        children: [
          cell([p([run(nextId(), { size: 20, color: GRAY })])], 1200, {
            shading: shade,
            verticalAlign: VerticalAlign.TOP,
          }),
          cell([p([run(bullet.text, { size: 20 })])], 7560, {
            shading: shade,
          }),
          cell(
            [
              p([
                run(priority, {
                  size: 20,
                  bold: true,
                  color: PRIORITY_COLOR[priority] || MID_BLUE,
                }),
              ]),
            ],
            600,
            { shading: shade },
          ),
        ],
      });
    }),
  ];
  section3Children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1200, 7560, 600],
      rows,
    }),
    p([
      run("P = Priority: M = Must Have, S = Should Have, C = Could Have", {
        size: 16,
        color: GRAY,
        italics: true,
      }),
    ]),
    ...spacer(1),
  );
}

section3Children.push(pageBreak());

// ── Section 4: Non-Functional Requirements ────────────────────────────────────
const nfrs = cfg.non_functional_requirements || {};

const section4 = [
  heading1("4. Non-Functional Requirements"),
  ...(Object.keys(nfrs).length > 0
    ? Object.entries(nfrs).flatMap(([category, items]) => [
        heading2(
          `4.${Object.keys(nfrs).indexOf(category) + 1} ${category}`,
        ),
        ...(items || []).map((item) => bulletPara(item)),
        ...spacer(1),
      ])
    : [
        p([
          run(
            "Non-functional requirements to be defined in collaboration with stakeholders.",
            { size: 22 },
          ),
        ]),
      ]),
  pageBreak(),
];

// ── Section 5: Constraints & Assumptions ──────────────────────────────────────
const section5 = [
  heading1("5. Constraints & Assumptions"),
  heading2("5.1 Constraints"),
  ...(
    cfg.constraints || [
      "Requirements were extracted from video content and may not be exhaustive.",
    ]
  ).map((c) => bulletPara(c)),
  ...spacer(1),
  heading2("5.2 Assumptions"),
  ...(
    cfg.assumptions || [
      "Users have basic familiarity with the application domain.",
    ]
  ).map((a) => bulletPara(a)),
  pageBreak(),
];

// ── Appendix A: Pipeline Metadata ─────────────────────────────────────────────
const priorityCounts = allBullets.reduce(
  (acc, b) => {
    const p = b.priority || "M";
    acc[p] = (acc[p] || 0) + 1;
    return acc;
  },
  { M: 0, S: 0, C: 0 },
);

const appendixA = [
  heading1("Appendix A: Analysis Metadata"),
  p([
    run(
      "The following metrics describe the automated extraction pipeline run that produced this document.",
      { size: 22 },
    ),
  ]),
  ...spacer(1),
  new Table({
    width: { size: 6000, type: WidthType.DXA },
    columnWidths: [3000, 3000],
    rows: [
      new TableRow({
        children: [labelCell("Metric", 3000), labelCell("Value", 3000)],
      }),
      ...[
        ["Total chunks processed", String(summary.total_chunks)],
        ["Successfully refined", String(summary.successfully_refined)],
        [
          "Cross-block duplicates removed",
          String(summary.duplicate_bullets_removed_cross_block),
        ],
        ["Total requirements extracted", String(allBullets.length)],
        [
          "Priority breakdown",
          `M: ${priorityCounts.M}  S: ${priorityCounts.S}  C: ${priorityCounts.C}`,
        ],
        ["Inference time", `${summary.inference_time_sec}s`],
        ["Judges refinement time", `${summary.judges_time_sec}s`],
        ["Total pipeline time", `${summary.total_time_sec}s`],
        ["Generated on", today],
      ].map(
        ([label, value]) =>
          new TableRow({
            children: [
              cell([p([run(label, { size: 20 })])], 3000),
              cell([p([run(value, { size: 20, bold: true })])], 3000),
            ],
          }),
      ),
    ],
  }),
];

// ── Assemble & write ──────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: {
          spacing: { before: 360, after: 120 },
          outlineLevel: 0,
          border: {
            bottom: {
              style: BorderStyle.SINGLE,
              size: 6,
              color: MID_BLUE,
              space: 4,
            },
          },
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: MID_BLUE },
        paragraph: {
          spacing: { before: 240, after: 80 },
          outlineLevel: 1,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: GRAY },
        paragraph: {
          spacing: { before: 160, after: 60 },
          outlineLevel: 2,
        },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: docHeader,
      footers: docFooter,
      children: [
        ...coverPage,
        new TableOfContents("Table of Contents", {
          hyperlink: true,
          headingStyleRange: "1-3",
          stylesWithLevels: [
            { styleName: "Heading1", level: 1 },
            { styleName: "Heading2", level: 2 },
          ],
        }),
        pageBreak(),
        ...section1,
        ...section2,
        ...section3Children,
        ...section4,
        ...section5,
        ...appendixA,
      ],
    },
  ],
});

fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });

Packer.toBuffer(doc)
  .then((buffer) => {
    fs.writeFileSync(OUTPUT_FILE, buffer);
    console.log(`✓ BRS written to ${OUTPUT_FILE}`);
    console.log(`  Project: ${cfg.project_name || "Unknown"}`);
    console.log(`  Categories: ${populatedCategories.length}`);
    console.log(`  Requirements: ${allBullets.length}`);
  })
  .catch((err) => {
    console.error("Failed to generate docx:", err);
    process.exit(1);
  });
