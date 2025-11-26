# Analysis of Drastic Changes in Any Ratios

## Overview
This analysis compares `any_percentage` values across three different type annotation approaches:
- **Untyped**: Files with no type annotations (Claude3-Sonnet 1st run)
- **Partial**: Files with partial type annotations (Claude3-Sonnet Partially Typed)
- **Fully**: Files with full type annotations (Claude3-Sonnet User Annotated)

## Dataset Statistics
- **Untyped files**: 1,407 files
- **Partial files**: 479 files  
- **Fully typed files**: 1,503 files

## Key Findings

### 1. Drastic Changes (30% threshold)
- **Total files with drastic changes**: 779 files
- This represents a significant portion of the dataset, indicating substantial differences in type annotation approaches

### 2. Pattern Analysis

#### Change Direction Trends:
- **Untyped → Partial**: 44 increases, 594 decreases
- **Untyped → Fully**: 269 increases, 203 decreases  
- **Partial → Fully**: 641 increases, 29 decreases

#### Key Observations:
1. **Most files show decreases** when moving from untyped to partial annotations
2. **Most files show increases** when moving from partial to fully typed annotations
3. **Files going from 0% to >20%**: 19 files
4. **Files going from >20% to 0%**: 7 files

### 3. Most Dramatic Changes (Top Examples)

#### Files with Extreme Changes:
1. **completion_8d3a60.py**: 
   - Untyped: 16.47% → Partial: 0% → Fully: 27.06%
   - Shows dramatic decrease in partial, then increase in fully typed

2. **sum__a10910.py**:
   - Untyped: 0% → Partial: 0% → Fully: 22.58%
   - Goes from no any types to significant any usage

3. **manifest_87d3d1.py**:
   - Untyped: 0% → Partial: 0% → Fully: 28.26%
   - Large file (230 slots) with significant any introduction

4. **base_c73d2b.py**:
   - Untyped: 7.61% → Partial: 0% → Fully: 23.91%
   - Shows 214% increase from untyped to fully typed

### 4. Critical Insights

#### Partial Typing Issues:
- Many files show **0% any_percentage** in partial typing, suggesting:
  - LLM-generated partial annotations may be avoiding `Any` types
  - Possible over-optimization or incomplete analysis
  - Files with 0 total slots in partial typing indicate missing data

#### Full Typing Behavior:
- **641 files show increases** from partial to fully typed
- This suggests that human annotators are more willing to use `Any` types when appropriate
- The increase indicates more realistic type annotation practices

#### Data Quality Concerns:
- Many files show `0/0` slots in partial typing, indicating:
  - Missing or incomplete data in the partial typing dataset
  - Potential issues with the LLM generation process
  - Need for data validation and cleaning

### 5. Recommendations

1. **Investigate Partial Typing Quality**: The high number of files with 0% in partial typing suggests potential issues with the LLM generation process.

2. **Validate Data Completeness**: Many files show missing slot data in partial typing, requiring data validation.

3. **Analyze Human vs LLM Patterns**: The significant differences between partial and fully typed annotations suggest different approaches to type annotation.

4. **Focus on High-Impact Files**: Files like `manifest_87d3d1.py` and `indexing_d02a25.py` show dramatic changes and warrant detailed investigation.

## Files Generated
- `drastic_changes_analysis.json`: Detailed results for 30% threshold
- `drastic_changes_100_percent.json`: Results for 100% threshold
- `analyze_any_ratio_changes.py`: Analysis script for future use

## Conclusion
The analysis reveals significant differences in type annotation approaches, with partial typing showing concerning patterns of missing data and potential over-optimization. The transition from partial to fully typed annotations shows more realistic type usage patterns, suggesting that human annotators provide more balanced type annotations.




