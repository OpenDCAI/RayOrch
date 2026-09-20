"""Deterministic document UDFs with nested Page and TableJob fan-out."""

from __future__ import annotations


class ParseDocuments:
    def run(self, documents):
        return [
            [
                {
                    "document": name,
                    "page": page,
                    "table_count": table_counts[page],
                }
                for page in range(len(table_counts))
            ]
            for name, table_counts in documents
        ]


class LayoutPages:
    def run(self, pages):
        return [
            {"page": page["page"], "layout": f"layout:{page['page']}"}
            for page in pages
        ]


class OcrPages:
    def run(self, pages, layouts):
        return [
            {
                "page": page["page"],
                "text": f"{page['document']}:{layout['layout']}",
            }
            for page, layout in zip(pages, layouts, strict=True)
        ]


class PostprocessPages:
    def run(self, pages, layouts, ocr):
        return [
            {
                **page,
                "layout": layout["layout"],
                "text": text["text"],
            }
            for page, layout, text in zip(pages, layouts, ocr, strict=True)
        ]


class ExpandTableJobs:
    def run(self, pages):
        return [
            [
                {
                    "document": page["document"],
                    "page": page["page"],
                    "table": table,
                }
                for table in range(page["table_count"])
            ]
            for page in pages
        ]


class TableCore:
    def run(self, jobs):
        return [{**job, "cells": job["table"] + 1} for job in jobs]


class ReducePage:
    def run(self, table_groups, pages):
        return [
            {
                "document": page["document"],
                "page": page["page"],
                "text": page["text"],
                "tables": tuple(table["table"] for table in tables),
            }
            for tables, page in zip(table_groups, pages, strict=True)
        ]


class ReduceDocument:
    def run(self, page_groups):
        return [
            {
                "document": pages[0]["document"],
                "pages": tuple(page["page"] for page in pages),
                "tables": tuple(page["tables"] for page in pages),
            }
            for pages in page_groups
        ]


__all__ = [
    "ExpandTableJobs",
    "LayoutPages",
    "OcrPages",
    "ParseDocuments",
    "PostprocessPages",
    "ReduceDocument",
    "ReducePage",
    "TableCore",
]
