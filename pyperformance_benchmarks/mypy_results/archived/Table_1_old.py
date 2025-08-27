import json

# Reload and recompute after reset
with open("mypy_results_deepseek.json") as f1, \
     open("mypy_results_gpt4O.json") as f2, \
     open("mypy_results_no_type.json") as f3, \
     open("mypy_results_o1_mini.json") as f4, \
     open("mypy_results_llama.json") as f5:
    deepseek = json.load(f1)
    gpt4o = json.load(f2)
    no_type = json.load(f3)
    o1_mini = json.load(f4)
    code_llama = json.load(f5)

def compute_stats(llm_data, no_type_data):
    unproc_files = len(llm_data)
    both_fail = 0
    both_pass = 0
    llm_only_fail = 0

    for filename in llm_data:
        llm_err = llm_data[filename].get("error_count", 0)
        no_type_err = no_type_data.get(filename, {}).get("error_count", 0)
        if llm_err > 0 and no_type_err > 0:
            both_fail += 1
        elif llm_err == 0 and no_type_err == 0:
            both_pass += 1
        elif llm_err > 0 and no_type_err == 0:
            llm_only_fail += 1

    llm_only_success_percent = 100 * (1-(llm_only_fail) / (both_pass+llm_only_fail))
    return unproc_files, both_fail, both_pass, llm_only_fail, llm_only_success_percent

# Compute stats
o1_stats = compute_stats(o1_mini, no_type)
deepseek_stats = compute_stats(deepseek, no_type)
gpt4o_stats = compute_stats(gpt4o, no_type)
code_llama_stats = compute_stats(code_llama, no_type)


print(gpt4o_stats)
print(o1_stats)
print(deepseek_stats)
print(code_llama_stats)
