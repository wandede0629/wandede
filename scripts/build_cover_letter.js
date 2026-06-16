// Cover letter for Journal of Energy Storage. Build: node build_cover_letter.js
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, AlignmentType } = require("docx");

const P = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 160 }, alignment: opts.align,
  children: [new TextRun({ text, size: 22, bold: opts.bold, italics: opts.italics })],
});
const B = (text) => new Paragraph({
  bullet: { level: 0 }, spacing: { after: 60 },
  children: [new TextRun({ text, size: 22 })],
});

const C = [];
C.push(P("16 June 2026", { after: 240 }));
C.push(P("To the Editors", { after: 0 }));
C.push(P("Journal of Energy Storage", { italics: true, after: 240 }));

C.push(P("Dear Editors,"));

C.push(P("We are pleased to submit our manuscript, “Interpretable and Uncertainty-Calibrated Lithium-Ion Battery Prognostics under Distribution Shift via Conformal Prediction and Physically Grounded Features,” for consideration as an original research article in Journal of Energy Storage."));

C.push(P("Reliable state-of-health (SOH) and remaining-useful-life (RUL) estimation is central to battery management, yet most data-driven models report only a single number, offer little physical insight, and are evaluated under random data splits that overstate real-world performance. Our work targets these gaps directly: we build an interpretable, uncertainty-calibrated SOH/RUL framework and evaluate it under strict cell-level, batch-aware protocols on public cycling data, so that no in-house cell cycling is required and every result is fully reproducible."));

C.push(P("We believe the manuscript fits the scope and readership of Journal of Energy Storage because it unites battery-health diagnostics, calibrated uncertainty for BMS decision support, and practical deployability. Its main contributions are:"));
C.push(B("SOH estimation on unseen cells at 1.26% RMSPE, with empirically calibrated split-conformal prediction intervals (90% coverage), and a single incremental-capacity feature carrying most of the model importance;"));
C.push(B("a deployable RUL pipeline that needs no capacity label at inference, and a partial 10–20% voltage window that retains near-full SOH accuracy, relevant to on-board estimation;"));
C.push(B("honest diagnostics that single-split studies hide: uncertainty-based active learning does not reliably beat random sampling, and naive cross-batch transfer collapses to negative R²;"));
C.push(B("adaptive conformal recalibration that keeps coverage near nominal under streaming distribution drift, where static intervals fail; and"));
C.push(B("a direct LFP-to-LCO cross-system transfer test that delineates the generalisation boundary — a frozen model collapses, whereas re-fitting the pipeline on a small labelled target set recovers point accuracy."));

C.push(P("This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. The study uses only publicly available datasets (Severson/MATR, NASA PCoE and CALCE) and involves no human or animal subjects. The analysis code and processing pipeline that reproduce all results are openly available at https://github.com/wandede0629/wandede."));

C.push(P("We hope you find the manuscript suitable for peer review, and we look forward to your response."));

C.push(P("Sincerely,", { after: 60 }));
C.push(P("Zhuohua Quan, on behalf of all authors", { after: 0 }));
C.push(P("Tianjin Lishen Battery Co., Ltd., Tianjin, China", { italics: true, after: 0 }));
C.push(P("quanzhuohua@lishen.com.cn"));

const doc = new Document({
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.env.OUT || "cover_letter.docx", buf);
  console.log("cover_letter.docx written,", buf.length, "bytes");
});
