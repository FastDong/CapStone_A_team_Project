from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
OUT_PATH = PAPER_DIR / "manuscript_ko_updated_with_figures.docx"
METRICS_DIR = ROOT / "ml" / "outputs" / "metrics"
PLOTS_DIR = ROOT / "ml" / "outputs" / "plots"
METADATA_DIR = ROOT / "ml" / "outputs" / "metadata"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format: {path}")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


class DocxBuilder:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.rels: list[str] = []
        self.media: list[tuple[str, bytes]] = []
        self.rel_idx = 1
        self.pic_idx = 1

    def _rpr(self, bold: bool = False, size: int = 22, italic: bool = False) -> str:
        bits = [
            '<w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/>',
            f'<w:sz w:val="{size}"/>',
            f'<w:szCs w:val="{size}"/>',
        ]
        if bold:
            bits.append("<w:b/>")
        if italic:
            bits.append("<w:i/>")
        return "<w:rPr>" + "".join(bits) + "</w:rPr>"

    def paragraph(self, text: str, *, bold: bool = False, size: int = 22, align: str = "left", italic: bool = False) -> None:
        jc = "" if align == "left" else f'<w:jc w:val="{align}"/>'
        runs: list[str] = []
        for i, line in enumerate(text.split("\n")):
            if i:
                runs.append("<w:br/>")
            runs.append(f'<w:t xml:space="preserve">{escape(line)}</w:t>')
        self.body.append(f'<w:p><w:pPr>{jc}</w:pPr><w:r>{self._rpr(bold=bold, size=size, italic=italic)}{"".join(runs)}</w:r></w:p>')

    def heading(self, text: str, level: int) -> None:
        size = 32 if level == 1 else 26 if level == 2 else 24
        self.paragraph(text, bold=True, size=size)

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows: list[list[str]]) -> None:
        grid = "".join('<w:gridCol w:w="1800"/>' for _ in rows[0])
        trs = []
        for ridx, row in enumerate(rows):
            cells = []
            for cell in row:
                bold = ridx == 0
                cells.append(
                    '<w:tc><w:tcPr><w:tcW w:w="1800" w:type="dxa"/></w:tcPr>'
                    f'<w:p><w:r>{self._rpr(bold=bold, size=19)}<w:t xml:space="preserve">{escape(str(cell))}</w:t></w:r></w:p></w:tc>'
                )
            trs.append("<w:tr>" + "".join(cells) + "</w:tr>")
        self.body.append(
            "<w:tbl><w:tblPr><w:tblBorders>"
            '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            f"</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
        )

    def image(self, path: Path, caption: str, max_width_inches: float = 2.4) -> None:
        width_px, height_px = read_png_size(path)
        cx = width_px * 9525
        cy = height_px * 9525
        max_cx = int(max_width_inches * 914400)
        if cx > max_cx:
            ratio = max_cx / cx
            cx = int(cx * ratio)
            cy = int(cy * ratio)
        rid = f"rId{self.rel_idx}"
        name = f"image{self.pic_idx}{path.suffix.lower()}"
        self.rel_idx += 1
        self.pic_idx += 1
        self.rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>')
        self.media.append((name, path.read_bytes()))
        drawing = (
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0"'
            ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
            ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
            ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{self.pic_idx + 10}" name="{escape(name)}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr>'
            f'<pic:cNvPr id="{self.pic_idx + 10}" name="{escape(name)}"/><pic:cNvPicPr/>'
            '</pic:nvPicPr><pic:blipFill>'
            f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
            '</pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
        )
        self.body.append(drawing)
        self.paragraph(caption, align="center", size=19, italic=True)

    def save(self, path: Path) -> None:
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"'
            ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
            ' xmlns:o="urn:schemas-microsoft-com:office:office"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
            ' xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
            ' xmlns:v="urn:schemas-microsoft-com:vml"'
            ' xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"'
            ' xmlns:w10="urn:schemas-microsoft-com:office:word"'
            ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'
            ' xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"'
            ' xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"'
            ' xmlns:wne="http://schemas.microsoft.com/office/2006/wordml"'
            ' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14">'
            "<w:body>" + "".join(self.body) +
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
            "</w:body></w:document>"
        )
        doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(self.rels) + "</Relationships>"
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        )
        app_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>OpenAI Codex</Application></Properties>'
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        core_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>MetS 2-stage framework manuscript</dc:title><dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("docProps/app.xml", app_xml)
            zf.writestr("docProps/core.xml", core_xml)
            for name, blob in self.media:
                zf.writestr(f"word/media/{name}", blob)


def pick_metrics(rows: list[dict], threshold_type: str) -> list[list[str]]:
    out = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Threshold"]]
    for row in rows:
        if row["threshold_type"] != threshold_type:
            continue
        out.append([
            row["model_name"],
            f"{float(row['accuracy']):.4f}",
            f"{float(row['precision']):.4f}",
            f"{float(row['recall']):.4f}",
            f"{float(row['f1']):.4f}",
            f"{float(row['roc_auc']):.4f}",
            f"{float(row['threshold_value']):.2f}",
        ])
    return out


def metric_row(rows: list[dict], model_name: str, threshold_type: str) -> dict:
    for row in rows:
        if row["model_name"] == model_name and row["threshold_type"] == threshold_type:
            return row
    raise KeyError((model_name, threshold_type))


def main() -> None:
    reg_rows = read_csv(METRICS_DIR / "regression_predict_metrics_best_models.csv")
    cls_rows = read_csv(METRICS_DIR / "classification_metrics_best_regression_models.csv")
    selected = json.loads((METADATA_DIR / "regression_best_model_selection.json").read_text(encoding="utf-8"))

    body_only_opt = metric_row(cls_rows, "body_only", "optimized")
    body_pred_default = metric_row(cls_rows, "body_plus_predicted", "default_0.5")
    body_pred_opt = metric_row(cls_rows, "body_plus_predicted", "optimized")
    body_actual_opt = metric_row(cls_rows, "body_plus_actual", "optimized")

    f1_gap = float(body_pred_opt["f1"]) - float(body_only_opt["f1"])
    auc_gap = float(body_pred_opt["roc_auc"]) - float(body_only_opt["roc_auc"])
    f1_rel = f1_gap / float(body_only_opt["f1"]) * 100.0
    auc_rel = auc_gap / float(body_only_opt["roc_auc"]) * 100.0
    f1_vs_actual = float(body_pred_opt["f1"]) / float(body_actual_opt["f1"]) * 100.0
    auc_vs_actual = float(body_pred_opt["roc_auc"]) / float(body_actual_opt["roc_auc"]) * 100.0

    builder = DocxBuilder()
    builder.paragraph("영양 및 신체계측 정보를 활용한 건강지표 predict 기반 2-stage framework를 통한 대사증후군 분류", bold=True, size=30, align="center")
    builder.paragraph("CapStone A Team", size=22, align="center")
    builder.paragraph(datetime.now().strftime("%Y-%m-%d"), size=20, align="center")

    builder.heading("초록", 1)
    builder.paragraph(
        "본 문서는 영양섭취 및 신체계측 정보만으로 건강지표를 먼저 predict하고, 그 predict 값을 활용해 대사증후군을 분류하는 "
        "2-stage framework의 최종 정리본이다. 1단계에서는 HE_glu, HE_sbp, HE_chol에 대해 여러 회귀모델을 비교한 뒤 "
        "각 target별 최고 성능 모델을 선택했고, 2단계에서는 body_only, body_plus_predicted, body_plus_actual 세 분류모델을 비교하였다. "
        f"최종적으로 body_plus_predicted는 기본 threshold에서 F1 {float(body_pred_default['f1']):.4f}, ROC-AUC {float(body_pred_default['roc_auc']):.4f}, "
        f"최적 threshold {float(body_pred_opt['threshold_value']):.2f} 적용 시 F1 {float(body_pred_opt['f1']):.4f}, ROC-AUC {float(body_pred_opt['roc_auc']):.4f}를 기록하였다. "
        "이는 신체지표만 사용하는 모델보다 높은 성능이며, 실제 건강지표를 사용하는 reference 모델에 근접한 수준이다."
    )

    builder.heading("1. 연구 목적", 1)
    builder.paragraph(
        "본 연구의 목적은 영양섭취 및 신체계측 정보만으로 생성한 predict 건강지표가 대사증후군 분류에 실제로 도움이 되는지 검증하는 데 있다. "
        "이를 위해 건강지표 예측 단계와 최종 분류 단계를 분리한 2-stage framework를 구성하였다."
    )

    builder.heading("2. 연구 절차", 1)
    builder.paragraph(
        "1) 데이터 로드 및 전처리\n"
        "2) train/test split 수행\n"
        "3) 건강지표별 회귀모델 후보 비교\n"
        "4) 각 target의 최고 성능 회귀모델 선택\n"
        "5) pred_glu, pred_sbp, pred_chol 생성\n"
        "6) body_only, body_plus_predicted, body_plus_actual 분류모델 학습\n"
        "7) 기본 threshold와 최적 threshold에서 성능 비교\n"
        "8) 선행연구와 성능 위치 비교"
    )

    builder.heading("3. 1단계 회귀모델", 1)
    builder.paragraph(
        "회귀 단계에서는 age, sex, HE_ht, HE_wt, HE_wc, WHtR, 영양섭취 변수와 파생 피처를 이용해 HE_glu, HE_sbp, HE_chol을 predict하였다. "
        "비교 모델은 xgboost_ref, lightgbm, extra_trees, random_forest, ridge였다. 세 건강지표에 동일한 모델을 고정하지 않고, "
        "각 target에서 가장 낮은 RMSE와 높은 R2를 보인 모델을 최종 선택하였다."
    )
    reg_table = [["Target", "Selected Model", "MAE", "RMSE", "R2"]]
    for row in reg_rows:
        reg_table.append([
            row["target"],
            row["selected_model"],
            f"{float(row['MAE']):.4f}",
            f"{float(row['RMSE']):.4f}",
            f"{float(row['R2']):.4f}",
        ])
    builder.table(reg_table)
    builder.paragraph("최종 선택 모델은 다음과 같다.")
    sel_table = [["Target", "Best Model"]]
    for target, model_name in selected.items():
        sel_table.append([target, model_name])
    builder.table(sel_table)
    builder.paragraph(
        "actual-predict scatter plot을 보면 세 target 모두에서 predict 값이 actual 값의 방향성을 일정 부분 추적한다. "
        "다만 극단값 구간에서는 평균 방향으로 수축되는 경향이 남아 있어, predict 건강지표는 실제값의 완전한 대체라기보다 "
        "분류단계에 투입 가능한 보조정보로 해석하는 것이 타당하다."
    )

    builder.paragraph("Regression summary figures", bold=True, size=24)
    builder.image(PLOTS_DIR / "regression" / "regression_model_benchmark_summary.png", "Figure. Cross-model regression benchmark by target", max_width_inches=5.5)
    builder.image(PLOTS_DIR / "regression" / "feature_importance_HE_glu.png", "Figure. HE_glu feature importance", max_width_inches=4.0)
    builder.image(PLOTS_DIR / "regression" / "feature_importance_HE_sbp.png", "Figure. HE_sbp feature importance", max_width_inches=4.0)
    builder.image(PLOTS_DIR / "regression" / "feature_importance_HE_chol.png", "Figure. HE_chol feature importance", max_width_inches=4.0)
    builder.paragraph("Detailed regression figures", bold=True, size=24)
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_glu.png", "Figure 1. HE_glu actual vs predict")
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_sbp.png", "Figure 2. HE_sbp actual vs predict")
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_chol.png", "Figure 3. HE_chol actual vs predict")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_glu.png", "Figure 4. HE_glu residual plot")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_sbp.png", "Figure 5. HE_sbp residual plot")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_chol.png", "Figure 6. HE_chol residual plot")

    builder.page_break()
    builder.heading("4. 2단계 분류모델", 1)
    builder.paragraph(
        "분류 단계에서는 XGBoost Classifier를 사용하여 세 가지 모델을 비교하였다. body_only는 신체지표만 사용하는 기준선 모델이고, "
        "body_plus_predicted는 1단계에서 생성한 predict 건강지표를 추가한 본 연구의 핵심 모델이며, body_plus_actual은 실제 건강지표를 사용하는 reference 모델이다."
    )
    builder.paragraph("기본 threshold 0.5 결과")
    builder.table(pick_metrics(cls_rows, "default_0.5"))
    builder.paragraph("최적 threshold 결과")
    builder.table(pick_metrics(cls_rows, "optimized"))
    builder.paragraph(
        f"최적 threshold 기준 body_plus_predicted는 F1 {float(body_pred_opt['f1']):.4f}, ROC-AUC {float(body_pred_opt['roc_auc']):.4f}를 기록하였다. "
        f"이는 body_only 대비 F1이 {f1_gap:+.4f}p, ROC-AUC가 {auc_gap:+.4f}p 높으며, 상대적으로는 F1 {f1_rel:.1f}%, ROC-AUC {auc_rel:.1f}% 향상이다. "
        f"또한 body_plus_actual 대비 F1의 {f1_vs_actual:.1f}%, ROC-AUC의 {auc_vs_actual:.1f}% 수준에 해당한다."
    )

    builder.paragraph("Classification summary figures", bold=True, size=24)
    builder.image(PLOTS_DIR / "classification" / "classification_default_summary.png", "Figure. Classification summary at default threshold", max_width_inches=4.8)
    builder.image(PLOTS_DIR / "classification" / "classification_optimized_summary.png", "Figure. Optimized threshold summary", max_width_inches=4.8)
    builder.image(PLOTS_DIR / "classification" / "roc_curve.png", "Figure 7. ROC curve")
    builder.image(PLOTS_DIR / "classification" / "pr_curve_body_only.png", "Figure. body_only precision-recall curve")
    builder.image(PLOTS_DIR / "classification" / "pr_curve_body_plus_predicted.png", "Figure. body_plus_predicted precision-recall curve")
    builder.image(PLOTS_DIR / "classification" / "pr_curve_body_plus_actual.png", "Figure. body_plus_actual precision-recall curve")
    builder.image(PLOTS_DIR / "classification" / "threshold_f1_body_only.png", "Figure. body_only threshold vs F1")
    builder.image(PLOTS_DIR / "classification" / "threshold_f1_body_plus_predicted.png", "Figure. body_plus_predicted threshold vs F1")
    builder.image(PLOTS_DIR / "classification" / "threshold_f1_body_plus_actual.png", "Figure. body_plus_actual threshold vs F1")
    builder.image(PLOTS_DIR / "classification" / "confusion_matrix_body_only.png", "Figure. body_only confusion matrix")
    builder.image(PLOTS_DIR / "classification" / "confusion_matrix_predicted.png", "Figure. body_plus_predicted confusion matrix")
    builder.image(PLOTS_DIR / "classification" / "confusion_matrix_actual.png", "Figure. body_plus_actual confusion matrix")
    builder.image(PLOTS_DIR / "classification" / "feature_importance_body_only_classifier.png", "Figure. body_only classifier feature importance")
    builder.image(PLOTS_DIR / "classification" / "feature_importance_predicted_classifier.png", "Figure. body_plus_predicted classifier feature importance")
    builder.image(PLOTS_DIR / "classification" / "feature_importance_actual_classifier.png", "Figure. body_plus_actual classifier feature importance")

    builder.heading("5. 선행연구와의 비교", 1)
    builder.paragraph(
        "본 연구의 body_plus_predicted ROC-AUC 0.8784는 Xu et al. (2022)이 비침습 anthropometric 기반 모델에서 보고한 AUC 0.880, "
        "외부검증 AUC 0.862와 유사하거나 다소 높은 수준이다. Shin et al. (2023)의 AUC 0.889보다는 약간 낮지만 비슷한 범위에 위치한다. "
        "반면 생화학 변수를 폭넓게 사용하는 biomarker-rich 연구의 AUC 0.940~0.954보다는 낮다. 이는 실제 혈액검사 없이 predict 건강지표를 활용한 "
        "실용적 스크리닝 구조로서의 의의에 초점을 맞춰 해석하는 것이 적절하다."
    )

    builder.heading("6. 결론", 1)
    builder.paragraph(
        "각 건강지표에서 최고 성능 회귀모델을 선택해 만든 predict 건강지표는 실제값을 완전히 대체하지는 못했지만, "
        "2단계 분류에서 body_only보다 더 높은 성능을 제공했다. 따라서 predict 건강지표는 MetS 분류를 위한 유의미한 보조정보로 작용한다고 해석할 수 있다."
    )
    builder.paragraph(
        "윤리적 고지: 본 문서의 결과는 건강관리 보조 및 선별 목적의 참고 정보이며, 의료적 진단, 치료, 처방을 대체하지 않는다.",
        italic=True,
        size=20,
    )

    out_path = OUT_PATH
    try:
        builder.save(out_path)
    except PermissionError:
        out_path = OUT_PATH.with_stem(OUT_PATH.stem + "_v2")
        builder.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
