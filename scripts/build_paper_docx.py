from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime
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
        bits = ['<w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/>',
                f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
        if bold:
            bits.append("<w:b/>")
        if italic:
            bits.append("<w:i/>")
        return "<w:rPr>" + "".join(bits) + "</w:rPr>"

    def paragraph(self, text: str, *, bold: bool = False, size: int = 22, align: str = "left", italic: bool = False) -> None:
        jc = "" if align == "left" else f'<w:jc w:val="{align}"/>'
        runs = []
        for i, line in enumerate(text.split("\n")):
            if i:
                runs.append("<w:br/>")
            runs.append(f"<w:t xml:space=\"preserve\">{escape(line)}</w:t>")
        self.body.append(f'<w:p><w:pPr>{jc}</w:pPr><w:r>{self._rpr(bold=bold, size=size, italic=italic)}{"".join(runs)}</w:r></w:p>')

    def heading(self, text: str, level: int) -> None:
        size = 32 if level == 1 else 26 if level == 2 else 24
        self.paragraph(text, bold=True, size=size)

    def page_break(self) -> None:
        self.body.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def table(self, rows: list[list[str]]) -> None:
        grid = "".join("<w:gridCol w:w=\"2400\"/>" for _ in rows[0])
        trs = []
        for ridx, row in enumerate(rows):
            cells = []
            for cell in row:
                bold = ridx == 0
                cells.append(
                    "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                    f'<w:p><w:r>{self._rpr(bold=bold, size=20)}<w:t xml:space="preserve">{escape(str(cell))}</w:t></w:r></w:p></w:tc>'
                )
            trs.append("<w:tr>" + "".join(cells) + "</w:tr>")
        self.body.append(
            "<w:tbl><w:tblPr><w:tblBorders>"
            "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
            "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
            "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
            "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"000000\"/>"
            "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"000000\"/>"
            "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"000000\"/>"
            f"</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
        )

    def image(self, path: Path, caption: str, max_width_inches: float = 6.0) -> None:
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
        self.paragraph(caption, align="center", size=20, italic=True)

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
            "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr>"
            "</w:body></w:document>"
        )
        doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(self.rels) + "</Relationships>"
        root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' \
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>' \
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>' \
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>' \
            "</Relationships>"
        content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' \
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' \
            '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>' \
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' \
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>' \
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>' \
            "</Types>"
        app_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>OpenAI Codex</Application></Properties>'
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        core_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>대사증후군 논문 정리</dc:title><dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'
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
            row["model_name"], f"{float(row['accuracy']):.4f}", f"{float(row['precision']):.4f}",
            f"{float(row['recall']):.4f}", f"{float(row['f1']):.4f}", f"{float(row['roc_auc']):.4f}",
            f"{float(row['threshold_value']):.2f}",
        ])
    return out


def main() -> None:
    reg_rows = read_csv(METRICS_DIR / "regression_baseline_vs_improved.csv")
    cls_rows = read_csv(METRICS_DIR / "classification_metrics.csv")
    selected = json.loads((METADATA_DIR / "regression_selected_configs.json").read_text(encoding="utf-8"))
    builder = DocxBuilder()
    builder.paragraph("영양 및 신체계측 정보를 활용한 건강지표 예측 기반 2단계 회귀-분류 프레임워크를 통한 대사증후군 분류", bold=True, size=32, align="center")
    builder.paragraph("CapStone A Team", size=22, align="center")
    builder.paragraph(datetime.now().strftime("%Y-%m-%d"), size=20, align="center")
    builder.heading("초록", 1)
    builder.paragraph("본 문서는 영양 및 신체계측 정보를 기반으로 건강지표를 예측한 뒤, 그 예측값을 활용하여 대사증후군을 분류한 2단계 회귀-분류 프레임워크의 최신 정리본이다. 회귀 단계에서는 타깃별 하이퍼파라미터 튜닝, 파생 피처 추가, 목적함수 비교, 타깃 변환, 그룹 기반 회귀 가능성을 비교하였다. 그 결과 HE_glu, HE_sbp, HE_chol 모두에서 기준선 대비 소폭 개선이 확인되었고, 분류 단계에서는 body_plus_predicted 모델이 실제 서비스 관점에서 가장 현실적인 대안으로 유지되었다.")
    builder.heading("1. 연구 개요", 1)
    builder.paragraph("본 연구는 임상검사 의존도가 높은 대사증후군 평가를 보조하기 위해, 비침습 정보만으로 건강지표를 먼저 추정하고 그 추정치를 다시 분류모델에 활용하는 구조를 검토하였다. 입력 변수는 연령, 성별, 키, 체중, 허리둘레, WHtR 및 영양섭취 지표로 구성되며, 최종 분류 라벨은 MetS_Label이다.")
    builder.heading("2. 회귀모델 개선 전략", 1)
    builder.paragraph("1) 타깃별 하이퍼파라미터를 별도로 탐색하였다.\n2) HE_glu와 HE_chol에 대해 log1p 변환 가능성을 검토하였다.\n3) reg:squarederror와 reg:pseudohubererror 목적함수를 비교하였다.\n4) BMI, 에너지 비율, 신체계측 상호작용 등 파생 피처를 추가하였다.\n5) global 모델과 sex-group-aware 회귀를 비교하였다.")
    builder.heading("3. 회귀모델 결과", 1)
    reg_table = [["Target", "Scenario", "MAE", "RMSE", "R2"]]
    for row in reg_rows:
        reg_table.append([row["target"], row["scenario"], f"{float(row['MAE']):.4f}", f"{float(row['RMSE']):.4f}", f"{float(row['R2']):.4f}"])
    builder.table(reg_table)
    builder.paragraph("최종 선택 조합은 다음과 같다.")
    sel_table = [["Target", "Group", "Transform", "Objective", "Param Set"]]
    for target, cfg in selected.items():
        sel_table.append([target, cfg["group_mode"], cfg["transform"], cfg["objective"], cfg["param_set"]])
    builder.table(sel_table)
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_glu.png", "그림 1. HE_glu actual vs predicted")
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_sbp.png", "그림 2. HE_sbp actual vs predicted")
    builder.image(PLOTS_DIR / "regression" / "actual_vs_predicted_HE_chol.png", "그림 3. HE_chol actual vs predicted")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_glu.png", "그림 4. HE_glu residual plot")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_sbp.png", "그림 5. HE_sbp residual plot")
    builder.image(PLOTS_DIR / "regression" / "residual_plot_HE_chol.png", "그림 6. HE_chol residual plot")
    builder.page_break()
    builder.heading("4. 분류모델 비교", 1)
    builder.paragraph("분류 단계에서는 동일한 train/test split에서 세 가지 입력 조합을 비교하였다. body_only는 신체지표만 사용하고, body_plus_predicted는 회귀 예측 건강지표를 추가하며, body_plus_actual은 실제 건강지표를 추가한 참조 모델이다.")
    builder.paragraph("기본 threshold 0.5 기준 결과")
    builder.table(pick_metrics(cls_rows, "default_0.5"))
    builder.paragraph("최적 threshold 적용 결과")
    builder.table(pick_metrics(cls_rows, "optimized"))
    builder.image(PLOTS_DIR / "classification" / "roc_curve.png", "그림 7. 분류모델 ROC curve")
    builder.image(PLOTS_DIR / "classification" / "threshold_f1_body_plus_predicted.png", "그림 8. body_plus_predicted threshold vs F1")
    builder.image(PLOTS_DIR / "classification" / "pr_curve_body_plus_predicted.png", "그림 9. body_plus_predicted precision-recall curve")
    builder.image(PLOTS_DIR / "classification" / "confusion_matrix_predicted.png", "그림 10. body_plus_predicted confusion matrix")
    builder.heading("5. 결론", 1)
    builder.paragraph("회귀모델의 설명력은 여전히 제한적이지만, 개선형 회귀 설정은 세 타깃 모두에서 기준선 대비 수치를 소폭 향상시켰다. 그 결과 body_plus_predicted 모델의 기본 F1은 0.6313, 최적 threshold 적용 F1은 0.6435로 향상되었다. 실제 건강지표를 사용하는 body_plus_actual이 최고 성능을 유지했지만, 실제 서비스 적용 관점에서는 예측 건강지표 기반 body_plus_predicted가 가장 현실적인 대안으로 판단된다.")
    builder.paragraph("윤리적 고지: 본 문서의 결과는 건강관리 보조 및 선별 목적의 참고 정보이며, 의료적 진단, 치료, 처방을 대체하지 않는다.", italic=True, size=20)
    builder.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
