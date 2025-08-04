from tree_sitter import Node as TsNode

def parse_text(code: bytes, node: TsNode) -> str:
    return code[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
