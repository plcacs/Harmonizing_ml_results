from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

candidates = [
    "unPremultipliedSkBitmap",
    "SkImageToPremultipliedAlpha",
    "convertUnpremultipliedBitmap",
    "premultipliedAlphaBuffer",
    "SkPremulColorToBitmap",
    "unPremulSkBitmapToPremul",
    "convertSkImageToPremul",
    "renderPremultipliedBitmap",
    "bufferAllocationFailure",
    "userAuthenticationToken",
    "memcpy_sse2_unaligned",
    "precomputedBufferLength",
]

results = []
for text in candidates:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    results.append((text, len(tokens), tokens))

# 按 subtoken 数量从多到少排序
results.sort(key=lambda x: x[1], reverse=True)

for text, n, tokens in results:
    print(f"{text:35s} -> {n:2d} subtokens: {tokens}")