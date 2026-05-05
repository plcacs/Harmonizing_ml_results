# AST Comparison Report: DeepSeek 2nd Run vs Untyped Baseline

## Summary Statistics

- **Total files compared**: 500
- **Files with structural changes**: 76
- **Files with parse errors**: 284
- **Files without baseline**: 2050
- **Percentage with changes**: 15.20%
- **Percentage with parse errors**: 56.80%
- **Percentage without baseline**: 80.39%

### Key Insights
- Out of 500 files, **76 (15.20%)** had structural differences
- **284** files could not be parsed (syntax errors in DeepSeek output)
- Most changes involve **control flow modifications**, **class additions**, or **parameter changes**

---

## ⚠️ Parse Errors - DeepSeek Generated Invalid Syntax

These files could not be analyzed because the generated code has syntax errors:

**_client_5a9e7d.py**

```
Error: '[' was never closed (<unknown>, line 435)
```

**alarm_control_panel_e80274.py**

```
Error: expected '(' (<unknown>, line 251)
```

**base_6106b5.py**

```
Error: expected '(' (<unknown>, line 398)
```

**base_9c14fc.py**

```
Error: '(' was never closed (<unknown>, line 286)
```

**boxplot_5aedc4.py**

```
Error: expected ':' (<unknown>, line 80)
```

**channels_8e6ab8.py**

```
Error: '(' was never closed (<unknown>, line 477)
```

**client_1e3c0a.py**

```
Error: expected ':' (<unknown>, line 377)
```

**conftest_ca46fe.py**

```
Error: '(' was never closed (<unknown>, line 357)
```

**httpclient_e36adc.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' (<unknown>, line 115)
```

**log_writer_c4397a.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' on line 44 (<unknown>, line 45)
```

**network_7ec1c6.py**

```
Error: '(' was never closed (<unknown>, line 371)
```

**network_manager_8c67f5.py**

```
Error: invalid syntax (<unknown>, line 443)
```

**params_3f7432.py**

```
Error: '(' was never closed (<unknown>, line 309)
```

**test_business_hour_6b0e7b.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' on line 119 (<unknown>, line 121)
```

**test_callables_9ff673.py**

```
Error: expected '(' (<unknown>, line 228)
```

**test_value_counts_77b3d1.py**

```
Error: expected an indented block after function definition on line 201 (<unknown>, line 201)
```

**train_test_b790e4.py**

```
Error: '(' was never closed (<unknown>, line 381)
```

**_models_dd672f.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' on line 373 (<unknown>, line 385)
```

**awsclient_860c67.py**

```
Error: expected '(' (<unknown>, line 367)
```

**base_tests_a1e90e.py**

```
Error: unterminated triple-quoted string literal (detected at line 445) (<unknown>, line 440)
```

**datetimes_c4b343.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' on line 27 (<unknown>, line 28)
```

**mocks_d2ab6d.py**

```
Error: '[' was never closed (<unknown>, line 43)
```

**smartcontracts_989d9a.py**

```
Error: expected '(' (<unknown>, line 403)
```

**test_appgraph_6f35e2.py**

```
Error: '(' was never closed (<unknown>, line 239)
```

**test_binance_2f9760.py**

```
Error: unterminated string literal (detected at line 112) (<unknown>, line 112)
```

**test_interface_94cb57.py**

```
Error: '(' was never closed (<unknown>, line 210)
```

**test_persistence_02086c.py**

```
Error: '[' was never closed (<unknown>, line 255)
```

**test_transition_77bcb3.py**

```
Error: '(' was never closed (<unknown>, line 213)
```

**testclient_6e1da5.py**

```
Error: '(' was never closed (<unknown>, line 346)
```

**augmented_lstm_c8e957.py**

```
Error: '(' was never closed (<unknown>, line 316)
```

**exposed_entities_a550ba.py**

```
Error: '(' was never closed (<unknown>, line 361)
```

**hive_66cde7.py**

```
Error: invalid syntax (<unknown>, line 338)
```

**imports_c4205d.py**

```
Error: '(' was never closed (<unknown>, line 377)
```

**methods_fdfe44.py**

```
Error: unexpected unindent (<unknown>, line 300)
```

**routing_66b713.py**

```
Error: '(' was never closed (<unknown>, line 377)
```

**test_conversion_928a39.py**

```
Error: unterminated string literal (detected at line 306) (<unknown>, line 306)
```

**test_dtypes_basic_4a9ebb.py**

```
Error: '(' was never closed (<unknown>, line 238)
```

**test_groupby_86d3bc.py**

```
Error: expected '(' (<unknown>, line 213)
```

**test_na_values_e137af.py**

```
Error: invalid syntax (<unknown>, line 43)
```

**test_nth_cbbcbb.py**

```
Error: '[' was never closed (<unknown>, line 224)
```

**test_plugins_493c2b.py**

```
Error: unterminated string literal (detected at line 265) (<unknown>, line 265)
```

**test_prefect_client_9ba189.py**

```
Error: '(' was never closed (<unknown>, line 381)
```

**test_rank_19ec93.py**

```
Error: '[' was never closed (<unknown>, line 219)
```

**test_reductions_87ff0d.py**

```
Error: unterminated string literal (detected at line 211) (<unknown>, line 211)
```

**base_9be364.py**

```
Error: '{' was never closed (<unknown>, line 469)
```

**confluent_3b2473.py**

```
Error: unterminated triple-quoted string literal (detected at line 402) (<unknown>, line 402)
```

**core_b5776a.py**

```
Error: '(' was never closed (<unknown>, line 63)
```

**deposits_09e7ec.py**

```
Error: '(' was never closed (<unknown>, line 261)
```

**expr_a7337f.py**

```
Error: '(' was never closed (<unknown>, line 422)
```

**google_config_4b4b29.py**

```
Error: invalid syntax (<unknown>, line 117)
```

**mediator_edd547.py**

```
Error: invalid syntax. Perhaps you forgot a comma? (<unknown>, line 45)
```

**screenshots_949c9a.py**

```
Error: invalid syntax (<unknown>, line 19)
```

**test_ccxt_compat_f96503.py**

```
Error: '(' was never closed (<unknown>, line 291)
```

**test_dtypes_c6ad71.py**

```
Error: expected ':' (<unknown>, line 379)
```

**test_hist_method_862ac9.py**

```
Error: '(' was never closed (<unknown>, line 359)
```

**test_indexing_e492e0.py**

```
Error: unterminated string literal (detected at line 277) (<unknown>, line 277)
```

**test_integration_events_f08b97.py**

```
Error: '(' was never closed (<unknown>, line 229)
```

**test_jinja_templated_action_39a881.py**

```
Error: '(' was never closed (<unknown>, line 430)
```

**test_join_48601a.py**

```
Error: unterminated string literal (detected at line 381) (<unknown>, line 381)
```

**test_mediatedtransfer_b63130.py**

```
Error: '(' was never closed (<unknown>, line 218)
```

**test_read_fwf_ec1bf4.py**

```
Error: '[' was never closed (<unknown>, line 124)
```

**test_state_changes_e74ee9.py**

```
Error: invalid syntax (<unknown>, line 300)
```

**test_utils_4b4e4d.py**

```
Error: '[' was never closed (<unknown>, line 175)
```

**value_05eb19.py**

```
Error: unterminated string literal (detected at line 450) (<unknown>, line 450)
```

**classes_29984e.py**

```
Error: unterminated triple-quoted string literal (detected at line 436) (<unknown>, line 435)
```

**collections_5f4a3a.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' (<unknown>, line 38)
```

**color_334856.py**

```
Error: '(' was never closed (<unknown>, line 306)
```

**deployer_c2beb1.py**

```
Error: '[' was never closed (<unknown>, line 407)
```

**fbeta_verbose_measure_test_a08d84.py**

```
Error: invalid syntax (<unknown>, line 270)
```

**join_merge_5c5786.py**

```
Error: unmatched ']' (<unknown>, line 63)
```

**multi_7dfe48.py**

```
Error: unterminated string literal (detected at line 297) (<unknown>, line 297)
```

**test_api_125d2c.py**

```
Error: unterminated string literal (detected at line 254) (<unknown>, line 254)
```

**test_arithmetics_2d6e2b.py**

```
Error: '(' was never closed (<unknown>, line 242)
```

**test_client_1103c5.py**

```
Error: '(' was never closed (<unknown>, line 252)
```

**test_filter_rewriting_c801f7.py**

```
Error: '[' was never closed (<unknown>, line 259)
```

**test_format_849ee9.py**

```
Error: unterminated string literal (detected at line 297) (<unknown>, line 297)
```

**test_freqai_interface_ca78dd.py**

```
Error: unterminated string literal (detected at line 203) (<unknown>, line 203)
```

**test_hyperopt_ac1e12.py**

```
Error: expected '(' (<unknown>, line 190)
```

**test_inference_83c9ac.py**

```
Error: expected an indented block after function definition on line 79 (<unknown>, line 82)
```

**test_process_withdrawal_request_9f973e.py**

```
Error: invalid syntax (<unknown>, line 249)
```

**test_remote_billing_3a9a00.py**

```
Error: '{' was never closed (<unknown>, line 278)
```

**test_template_ceacd4.py**

```
Error: '(' was never closed (<unknown>, line 288)
```

**transport_58bb8a.py**

```
Error: '(' was never closed (<unknown>, line 330)
```

**__init___344c2e.py**

```
Error: invalid syntax (<unknown>, line 231)
```

**__init___7ab01a.py**

```
Error: unterminated triple-quoted string literal (detected at line 466) (<unknown>, line 466)
```

**base_c73d2b.py**

```
Error: invalid syntax (<unknown>, line 116)
```

**celery_tests_fc4a66.py**

```
Error: unexpected unindent (<unknown>, line 241)
```

**conftest_641a68.py**

```
Error: '(' was never closed (<unknown>, line 298)
```

**dataprovider_a0d211.py**

```
Error: expected an indented block after 'if' statement on line 327 (<unknown>, line 328)
```

**event_0577a3.py**

```
Error: invalid syntax (<unknown>, line 356)
```

**event_schema_4d75fe.py**

```
Error: unterminated string literal (detected at line 287) (<unknown>, line 287)
```

**manifest_47c52e.py**

```
Error: '{' was never closed (<unknown>, line 370)
```

**runner_7722d1.py**

```
Error: unterminated f-string literal (detected at line 430) (<unknown>, line 430)
```

**sentiment_analysis_suite_5e67f5.py**

```
Error: unterminated string literal (detected at line 163) (<unknown>, line 163)
```

**template_entity_f0503d.py**

```
Error: '{' was never closed (<unknown>, line 376)
```

**test_astype_5e9cc9.py**

```
Error: invalid syntax (<unknown>, line 294)
```

**test_backtesting_c617d3.py**

```
Error: unterminated string literal (detected at line 314) (<unknown>, line 314)
```

**test_base_fc0d64.py**

```
Error: invalid syntax (<unknown>, line 315)
```

**test_cut_a5ac0a.py**

```
Error: invalid syntax (<unknown>, line 388)
```

**test_datetime_index_0892c0.py**

```
Error: expected an indented block after function definition on line 247 (<unknown>, line 248)
```

**test_http_parser_74f389.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' (<unknown>, line 38)
```

**test_series_apply_554287.py**

```
Error: '[' was never closed (<unknown>, line 297)
```

**test_sort_index_15410c.py**

```
Error: '[' was never closed (<unknown>, line 231)
```

**test_xs_1bb8c5.py**

```
Error: unterminated string literal (detected at line 382) (<unknown>, line 382)
```

**__init___14fc3d.py**

```
Error: expected an indented block after 'except' statement on line 349 (<unknown>, line 350)
```

**attestations_9893da.py**

```
Error: '(' was never closed (<unknown>, line 273)
```

**beam_search_test_a8133f.py**

```
Error: expected ':' (<unknown>, line 189)
```

**cloud_storage_b743ea.py**

```
Error: unterminated triple-quoted string literal (detected at line 408) (<unknown>, line 406)
```

**common_ee0adc.py**

```
Error: unterminated triple-quoted string literal (detected at line 497) (<unknown>, line 490)
```

**context_54b9a9.py**

```
Error: invalid syntax (<unknown>, line 227)
```

**converter_e14395.py**

```
Error: '(' was never closed (<unknown>, line 148)
```

**dockerutils_6badca.py**

```
Error: unterminated triple-quoted string literal (detected at line 420) (<unknown>, line 419)
```

**fairness_metrics_fcc9f2.py**

```
Error: unterminated string literal (detected at line 256) (<unknown>, line 256)
```

**packager_d0d7be.py**

```
Error: '(' was never closed (<unknown>, line 322)
```

**realm_settings_e9d618.py**

```
Error: expected '(' (<unknown>, line 244)
```

**sas7bdat_f3d8c2.py**

```
Error: invalid syntax (<unknown>, line 372)
```

**schema_yaml_readers_9dabb0.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**server_057ee6.py**

```
Error: expected 'except' or 'finally' block (<unknown>, line 300)
```

**test_deps_61aa2a.py**

```
Error: unterminated string literal (detected at line 275) (<unknown>, line 275)
```

**test_expanding_5d3564.py**

```
Error: invalid syntax (<unknown>, line 200)
```

**test_helpers_38220e.py**

```
Error: '(' was never closed (<unknown>, line 338)
```

**test_indexing_b72bb8.py**

```
Error: '(' was never closed (<unknown>, line 224)
```

**test_raises_891508.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' (<unknown>, line 34)
```

**test_setops_9f3297.py**

```
Error: unterminated string literal (detected at line 325) (<unknown>, line 325)
```

**test_usecols_basic_f18d3b.py**

```
Error: expected '(' (<unknown>, line 214)
```

**test_utils_014a22.py**

```
Error: '(' was never closed (<unknown>, line 379)
```

**utils_1a8fd5.py**

```
Error: invalid syntax (<unknown>, line 281)
```

**test_constructors_4c47aa.py**

```
Error: unexpected unindent (<unknown>, line 289)
```

**test_core_9112bf.py**

```
Error: '(' was never closed (<unknown>, line 392)
```

**test_edge_d080a8.py**

```
Error: unterminated string literal (detected at line 170) (<unknown>, line 170)
```

**test_flow_run_d3356c.py**

```
Error: unterminated string literal (detected at line 195) (<unknown>, line 195)
```

**test_index_d51bbf.py**

```
Error: '(' was never closed (<unknown>, line 229)
```

**test_indexing_235f4f.py**

```
Error: invalid syntax. Perhaps you forgot a comma? (<unknown>, line 208)
```

**test_merge_asof_031054.py**

```
Error: unterminated string literal (detected at line 64) (<unknown>, line 64)
```

**test_query_eval_130e95.py**

```
Error: '(' was never closed (<unknown>, line 299)
```

**test_reflection_3e7847.py**

```
Error: invalid syntax (<unknown>, line 341)
```

**test_rolling_741e1f.py**

```
Error: '[' was never closed (<unknown>, line 191)
```

**test_to_csv_2a1c22.py**

```
Error: '(' was never closed (<unknown>, line 169)
```

**test_win_type_fcca97.py**

```
Error: '[' was never closed (<unknown>, line 184)
```

**from_params_test_811442.py**

```
Error: unterminated string literal (detected at line 403) (<unknown>, line 403)
```

**indicators_960952.py**

```
Error: expected '(' (<unknown>, line 281)
```

**initiator_fb22e8.py**

```
Error: '(' was never closed (<unknown>, line 270)
```

**media_player_468e90.py**

```
Error: expected ':' (<unknown>, line 360)
```

**string__8c20b1.py**

```
Error: '(' was never closed (<unknown>, line 384)
```

**test_aggregate_1eee57.py**

```
Error: unterminated string literal (detected at line 281) (<unknown>, line 281)
```

**test_alerts_3e2087.py**

```
Error: '(' was never closed (<unknown>, line 329)
```

**test_cloud_storage_1d2f60.py**

```
Error: '(' was never closed (<unknown>, line 83)
```

**test_html_cda654.py**

```
Error: '(' was never closed (<unknown>, line 264)
```

**test_inference_f5d827.py**

```
Error: invalid syntax (<unknown>, line 183)
```

**test_rocksdb_e6756b.py**

```
Error: '{' was never closed (<unknown>, line 304)
```

**_compat_bfcdb3.py**

```
Error: '[' was never closed (<unknown>, line 91)
```

**base_value_caa0cc.py**

```
Error: '(' was never closed (<unknown>, line 422)
```

**instance_410c65.py**

```
Error: invalid syntax (<unknown>, line 111)
```

**object_array_68abb9.py**

```
Error: expected ':' (<unknown>, line 405)
```

**run_b777ab.py**

```
Error: expected ':' (<unknown>, line 323)
```

**stdlib_4fd1ea.py**

```
Error: invalid syntax (<unknown>, line 335)
```

**test_base_397a75.py**

```
Error: unterminated string literal (detected at line 340) (<unknown>, line 340)
```

**test_cli_3a2b0d.py**

```
Error: '(' was never closed (<unknown>, line 320)
```

**test_frame_color_e225c0.py**

```
Error: '(' was never closed (<unknown>, line 250)
```

**test_kedro_data_catalog_4cfadd.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' on line 246 (<unknown>, line 248)
```

**test_resample_api_f0cdf1.py**

```
Error: '[' was never closed (<unknown>, line 281)
```

**test_rpc_0215ce.py**

```
Error: unterminated string literal (detected at line 185) (<unknown>, line 185)
```

**test_series_5f79c1.py**

```
Error: unterminated string literal (detected at line 299) (<unknown>, line 299)
```

**test_stateful_cf2044.py**

```
Error: '(' was never closed (<unknown>, line 427)
```

**test_token_network_03ccab.py**

```
Error: '(' was never closed (<unknown>, line 171)
```

**test_wrappers_62b3f0.py**

```
Error: '[' was never closed (<unknown>, line 351)
```

**users_e2ac2d.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '[' (<unknown>, line 270)
```

**conftest_445a11.py**

```
Error: unterminated string literal (detected at line 376) (<unknown>, line 376)
```

**fbeta_multi_label_measure_test_75eeda.py**

```
Error: '[' was never closed (<unknown>, line 261)
```

**kedro_data_catalog_148318.py**

```
Error: unterminated triple-quoted string literal (detected at line 394) (<unknown>, line 392)
```

**parquet_a441a6.py**

```
Error: '[' was never closed (<unknown>, line 445)
```

**test_cli_4e6ec5.py**

```
Error: '(' was never closed (<unknown>, line 60)
```

**test_context_0bb4b5.py**

```
Error: '(' was never closed (<unknown>, line 414)
```

**test_datetimelike_0b6f81.py**

```
Error: '(' was never closed (<unknown>, line 258)
```

**test_eth1_chaindb_433a8a.py**

```
Error: '(' was never closed (<unknown>, line 78)
```

**test_format_57a8f1.py**

```
Error: unterminated string literal (detected at line 211) (<unknown>, line 211)
```

**test_ipython_b46143.py**

```
Error: '(' was never closed (<unknown>, line 420)
```

**test_matrix_transport_fc1c06.py**

```
Error: '(' was never closed (<unknown>, line 311)
```

**test_nodes_c6ed78.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' (<unknown>, line 93)
```

**test_replace_fc3342.py**

```
Error: unterminated string literal (detected at line 302) (<unknown>, line 302)
```

**test_rest_9e5773.py**

```
Error: '(' was never closed (<unknown>, line 243)
```

**test_settlement_0a3fe7.py**

```
Error: unterminated triple-quoted string literal (detected at line 212) (<unknown>, line 208)
```

**test_stack_unstack_b54311.py**

```
Error: unterminated string literal (detected at line 209) (<unknown>, line 209)
```

**__init___4ea406.py**

```
Error: invalid syntax. Perhaps you forgot a comma? (<unknown>, line 46)
```

**base_03e7f5.py**

```
Error: '{' was never closed (<unknown>, line 234)
```

**datastructures_0a7b11.py**

```
Error: invalid syntax (<unknown>, line 445)
```

**header_990b83.py**

```
Error: '(' was never closed (<unknown>, line 334)
```

**randomized_block_tests_8fb192.py**

```
Error: '(' was never closed (<unknown>, line 292)
```

**test_abstract_0be2c2.py**

```
Error: '[' was never closed (<unknown>, line 114)
```

**test_api_0e252b.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' (<unknown>, line 353)
```

**test_arithmetic_ef3dfe.py**

```
Error: '(' was never closed (<unknown>, line 293)
```

**test_base_3a84cb.py**

```
Error: '(' was never closed (<unknown>, line 321)
```

**test_base_worker_f71266.py**

```
Error: expected ':' (<unknown>, line 236)
```

**test_constructors_c6bb5c.py**

```
Error: unterminated string literal (detected at line 284) (<unknown>, line 284)
```

**test_consumer_4478ae.py**

```
Error: unterminated string literal (detected at line 416) (<unknown>, line 416)
```

**test_datetimes_630624.py**

```
Error: '[' was never closed (<unknown>, line 314)
```

**test_frame_60c296.py**

```
Error: '(' was never closed (<unknown>, line 248)
```

**test_frame_apply_391b73.py**

```
Error: '(' was never closed (<unknown>, line 311)
```

**test_indexing_36b06e.py**

```
Error: '[' was never closed (<unknown>, line 233)
```

**test_numba_c3554f.py**

```
Error: '(' was never closed (<unknown>, line 232)
```

**test_ops_on_diff_frames_1c24fd.py**

```
Error: expected ':' (<unknown>, line 228)
```

**test_selector_bbc47d.py**

```
Error: expected '(' (<unknown>, line 134)
```

**test_sort_values_87e12b.py**

```
Error: '(' was never closed (<unknown>, line 225)
```

**user_deposit_72a164.py**

```
Error: unmatched ')' (<unknown>, line 69)
```

**wrappers_34ca66.py**

```
Error: '(' was never closed (<unknown>, line 458)
```

**artifacts_099b0a.py**

```
Error: '(' was never closed (<unknown>, line 447)
```

**label_model_609423.py**

```
Error: '[' was never closed (<unknown>, line 311)
```

**notify_014492.py**

```
Error: unterminated string literal (detected at line 462) (<unknown>, line 462)
```

**switch_24834b.py**

```
Error: invalid syntax (<unknown>, line 312)
```

**tcpclient_test_1d9a53.py**

```
Error: '(' was never closed (<unknown>, line 364)
```

**test_converter_f508e6.py**

```
Error: '[' was never closed (<unknown>, line 118)
```

**test_flags_92e1da.py**

```
Error: invalid syntax (<unknown>, line 289)
```

**test_frame_plot_matplotlib_9669ca.py**

```
Error: '(' was never closed (<unknown>, line 270)
```

**test_json_7e4832.py**

```
Error: unterminated string literal (detected at line 312) (<unknown>, line 312)
```

**test_modular_pipeline_544bcc.py**

```
Error: invalid syntax (<unknown>, line 211)
```

**test_multi_209bc7.py**

```
Error: unterminated string literal (detected at line 184) (<unknown>, line 184)
```

**test_offsets_595cf3.py**

```
Error: expected an indented block after 'if' statement on line 202 (<unknown>, line 202)
```

**test_pivot_62e6c1.py**

```
Error: '[' was never closed (<unknown>, line 212)
```

**test_wallets_fe5a94.py**

```
Error: unterminated string literal (detected at line 171) (<unknown>, line 171)
```

**appgraph_e5732d.py**

```
Error: '(' was never closed (<unknown>, line 451)
```

**commands_tests_eb6597.py**

```
Error: unterminated string literal (detected at line 503) (<unknown>, line 503)
```

**cover_ccbdb1.py**

```
Error: expected ':' (<unknown>, line 400)
```

**fbeta_measure_test_d2fc57.py**

```
Error: expected an indented block after function definition on line 283 (<unknown>, line 283)
```

**media_source_5cb2e6.py**

```
Error: unterminated triple-quoted string literal (detected at line 334) (<unknown>, line 334)
```

**s3_48c5ab.py**

```
Error: invalid syntax (<unknown>, line 411)
```

**test_alt_backend_8ffa2d.py**

```
Error: '(' was never closed (<unknown>, line 389)
```

**test_constraints_cdcb72.py**

```
Error: unterminated string literal (detected at line 269) (<unknown>, line 269)
```

**test_gen_data_e67a06.py**

```
Error: unterminated string literal (detected at line 270) (<unknown>, line 270)
```

**test_pairwise_9290e6.py**

```
Error: '(' was never closed (<unknown>, line 207)
```

**test_partial_86bf04.py**

```
Error: '(' was never closed (<unknown>, line 293)
```

**test_scalar_compat_34fd8d.py**

```
Error: '[' was never closed (<unknown>, line 261)
```

**test_store_f376cd.py**

```
Error: unterminated string literal (detected at line 246) (<unknown>, line 246)
```

**test_validation_2c8d12.py**

```
Error: expected an indented block after function definition on line 181 (<unknown>, line 181)
```

**util_e9973a.py**

```
Error: '(' was never closed (<unknown>, line 296)
```

**__init___db92f1.py**

```
Error: parameter without a default follows parameter with a default (<unknown>, line 301)
```

**base_b13513.py**

```
Error: invalid syntax. Perhaps you forgot a comma? (<unknown>, line 110)
```

**forms_6e49d8.py**

```
Error: expected ':' (<unknown>, line 353)
```

**initiator_manager_26765c.py**

```
Error: '(' was never closed (<unknown>, line 389)
```

**jinja_context_test_42ca63.py**

```
Error: unterminated string literal (detected at line 310) (<unknown>, line 310)
```

**test_decimal_c85894.py**

```
Error: '(' was never closed (<unknown>, line 324)
```

**test_entity_registry_95c378.py**

```
Error: '[' was never closed (<unknown>, line 193)
```

**test_history_818171.py**

```
Error: '(' was never closed (<unknown>, line 193)
```

**test_interval_3fb811.py**

```
Error: '(' was never closed (<unknown>, line 290)
```

**test_logging_b7910e.py**

```
Error: invalid syntax (<unknown>, line 284)
```

**test_matrix_transport_afcccc.py**

```
Error: '(' was never closed (<unknown>, line 355)
```

**test_rank_c4ccb1.py**

```
Error: '(' was never closed (<unknown>, line 213)
```

**test_reductions_08a693.py**

```
Error: invalid syntax (<unknown>, line 63)
```

**test_reindex_54a65d.py**

```
Error: '{' was never closed (<unknown>, line 208)
```

**test_rolling_functions_251098.py**

```
Error: '(' was never closed (<unknown>, line 150)
```

**test_round_trip_b8b484.py**

```
Error: '(' was never closed (<unknown>, line 254)
```

**test_s3_fad4be.py**

```
Error: expected '(' (<unknown>, line 285)
```

**test_set_index_3c0de8.py**

```
Error: unterminated string literal (detected at line 236) (<unknown>, line 236)
```

**test_timedelta64_f628bf.py**

```
Error: unterminated string literal (detected at line 293) (<unknown>, line 293)
```

**test_to_latex_4e44c5.py**

```
Error: unterminated string literal (detected at line 223) (<unknown>, line 223)
```

**test_to_numeric_199e51.py**

```
Error: unterminated string literal (detected at line 284) (<unknown>, line 284)
```

**cache_policies_2cc3d9.py**

```
Error: '[' was never closed (<unknown>, line 100)
```

**clienttrader_5e3b99.py**

```
Error: expected an indented block after 'except' statement on line 382 (<unknown>, line 382)
```

**coordinator_e4c72e.py**

```
Error: invalid syntax (<unknown>, line 288)
```

**retention_81eb85.py**

```
Error: unterminated string literal (detected at line 316) (<unknown>, line 316)
```

**test_common_basic_a8eb4c.py**

```
Error: unterminated string literal (detected at line 185) (<unknown>, line 185)
```

**test_converter_orderflow_7dad6d.py**

```
Error: '[' was never closed (<unknown>, line 211)
```

**test_datetimelike_db9f17.py**

```
Error: invalid syntax (<unknown>, line 310)
```

**test_libsparse_7cdef7.py**

```
Error: '(' was never closed (<unknown>, line 270)
```

**test_opcodes_be536f.py**

```
Error: unterminated string literal (detected at line 126) (<unknown>, line 126)
```

**test_rpc_telegram_deaea2.py**

```
Error: unterminated string literal (detected at line 298) (<unknown>, line 298)
```

**test_setitem_a50d5b.py**

```
Error: '[' was never closed (<unknown>, line 264)
```

**test_sparse_d759ae.py**

```
Error: expected ':' (<unknown>, line 302)
```

**test_sql_f4958a.py**

```
Error: expected ':' (<unknown>, line 327)
```

**test_strings_4cd98f.py**

```
Error: unmatched ')' (<unknown>, line 119)
```

**util_test_ea6c40.py**

```
Error: '(' was never closed (<unknown>, line 168)
```

**email_mirror_590460.py**

```
Error: '(' was never closed (<unknown>, line 342)
```

**iterable_71c277.py**

```
Error: expected an indented block after 'for' statement on line 380 (<unknown>, line 381)
```

**project_7870ca.py**

```
Error: '(' was never closed (<unknown>, line 275)
```

**string_arrow_ce4e9d.py**

```
Error: invalid syntax (<unknown>, line 75)
```

**stubs_b63ee1.py**

```
Error: invalid syntax (<unknown>, line 79)
```

**test_analytics_c6af0b.py**

```
Error: closing parenthesis ']' does not match opening parenthesis '(' (<unknown>, line 57)
```

**test_arithmetic_f016de.py**

```
Error: '(' was never closed (<unknown>, line 281)
```

**test_arrays_9fefe1.py**

```
Error: unmatched ')' (<unknown>, line 187)
```

**test_core_515237.py**

```
Error: '(' was never closed (<unknown>, line 344)
```

**test_sas7bdat_0ac638.py**

```
Error: unterminated string literal (detected at line 251) (<unknown>, line 251)
```

**test_sorting_c07df0.py**

```
Error: '(' was never closed (<unknown>, line 31)
```

**test_stata_db4b21.py**

```
Error: '[' was never closed (<unknown>, line 357)
```

**test_to_html_24b1e4.py**

```
Error: unterminated string literal (detected at line 238) (<unknown>, line 238)
```

**test_websocket_parser_3e2e83.py**

```
Error: '(' was never closed (<unknown>, line 317)
```

---

## 📋 Files Without Baseline (2050 total)

These files exist in DeepSeek output but have no matching file in the 500_untyped_files baseline:

- `__init___4bc6fc.py` (Folder 1)
- `__init___892eae.py` (Folder 1)
- `__init___b65f90.py` (Folder 1)
- `__init___bc9a6b.py` (Folder 1)
- `_api_8facbb.py` (Folder 1)
- `_base_2ce4a7.py` (Folder 1)
- `_core_627b18.py` (Folder 1)
- `_json_57f411.py` (Folder 1)
- `agents_3f7d1d.py` (Folder 1)
- `api_tests_76e6db.py` (Folder 1)
- `applications_016968.py` (Folder 1)
- `asserters_700def.py` (Folder 1)
- `base_4b5828.py` (Folder 1)
- `base_ce86d4.py` (Folder 1)
- `blocks_a65ab4.py` (Folder 1)
- `blocks_abf4af.py` (Folder 1)
- `cache_backend_1dcc35.py` (Folder 1)
- `callback_4e27f8.py` (Folder 1)
- `case_2f1853.py` (Folder 1)
- `channels_7d2bb8.py` (Folder 1)
- `checkpoint_decoder_4db09b.py` (Folder 1)
- `cli_800977.py` (Folder 1)
- `client_8529a4.py` (Folder 1)
- `client_87f577.py` (Folder 1)
- `client_reqrep_ca32d6.py` (Folder 1)
- `climate_588757.py` (Folder 1)
- `conftest_976174.py` (Folder 1)
- `connectionpool_8dc770.py` (Folder 1)
- `connectionpool_96094f.py` (Folder 1)
- `core_336d4b.py` (Folder 1)
- `core_8f014a.py` (Folder 1)
- `core_a86387.py` (Folder 1)
- `core_c55873.py` (Folder 1)
- `create_user_36a545.py` (Folder 1)
- `create_user_5beae0.py` (Folder 1)
- `data_io_d2e959.py` (Folder 1)
- `dataclasses_968ced.py` (Folder 1)
- `datetimes_224965.py` (Folder 1)
- `default_bb0007.py` (Folder 1)
- `embedding_8c6937.py` (Folder 1)
- `execution_context_61aa2e.py` (Folder 1)
- `export_6707ca.py` (Folder 1)
- `fields_c8e6d9.py` (Folder 1)
- `fields_deface.py` (Folder 1)
- `flows_b84e48.py` (Folder 1)
- `gradient_descent_trainer_096f6d.py` (Folder 1)
- `headers_cc92a5.py` (Folder 1)
- `helpers_b52294.py` (Folder 1)
- `http_parser_0d16fa.py` (Folder 1)
- `import_util_c23496.py` (Folder 1)
- `inference_7e0d57.py` (Folder 1)
- `influence_interpreter_a2577a.py` (Folder 1)
- `instaloader_dceea7.py` (Folder 1)
- `integrations_55ac12.py` (Folder 1)
- `interface_c3b648.py` (Folder 1)
- `issue_registry_f55ae7.py` (Folder 1)
- `json-log-to-html_6abdf7.py` (Folder 1)
- `json_7b3477.py` (Folder 1)
- `legacy_d0ef53.py` (Folder 1)
- `light_41b89c.py` (Folder 1)
- `main_b9b094.py` (Folder 1)
- `manager_2630e6.py` (Folder 1)
- `manifest_87d3d1.py` (Folder 1)
- `matplotlib_5964e7.py` (Folder 1)
- `mattermost_5a1f15.py` (Folder 1)
- `missing_3e0a9c.py` (Folder 1)
- `model_card_7b7f62.py` (Folder 1)
- `multiprocess_data_loader_8b38d9.py` (Folder 1)
- `namespace_67e527.py` (Folder 1)
- `network_55c06b.py` (Folder 1)
- `networks_4fcfb8.py` (Folder 1)
- `notification_data_d3e584.py` (Folder 1)
- `param_functions_a348d9.py` (Folder 1)
- `params_067b56.py` (Folder 1)
- `pretrained_transformer_embedder_dde022.py` (Folder 1)
- `project_72b75e.py` (Folder 1)
- `pytables_a9636c.py` (Folder 1)
- `query_object_6550f7.py` (Folder 1)
- `readers_9a2654.py` (Folder 1)
- `record_c0e555.py` (Folder 1)
- `resample_616b7d.py` (Folder 1)
- `response_563505.py` (Folder 1)
- `rocketchat_52a9b8.py` (Folder 1)
- `root_model_c9296c.py` (Folder 1)
- `routing_32e1b9.py` (Folder 1)
- `runner_ecb0c0.py` (Folder 1)
- `sensor_c48c5b.py` (Folder 1)
- `series_26b25d.py` (Folder 1)
- `series_e95f35.py` (Folder 1)
- `settings_902e0a.py` (Folder 1)
- `slack_98777b.py` (Folder 1)
- `stata_4ae9e0.py` (Folder 1)
- `stats_691b29.py` (Folder 1)
- `stores_aa1665.py` (Folder 1)
- `strategy_test_v3_251abc.py` (Folder 1)
- `style_5e569b.py` (Folder 1)
- `support_f17c76.py` (Folder 1)
- `switch_67758e.py` (Folder 1)
- `t5_f23eca.py` (Folder 1)
- `tasks_d21306.py` (Folder 1)
- `test_aiokafka_784fbc.py` (Folder 1)
- `test_app_e3d659.py` (Folder 1)
- `test_classes_e83569.py` (Folder 1)
- `test_exchange_2ba68d.py` (Folder 1)
- `test_providers_4601ba.py` (Folder 1)
- `test_utils_121373.py` (Folder 1)
- `topics_3a0e29.py` (Folder 1)
- `topics_fc6164.py` (Folder 1)
- `training_45f057.py` (Folder 1)
- `transactions_ca9596.py` (Folder 1)
- `utils_3fa178.py` (Folder 1)
- `utils_9c5ea6.py` (Folder 1)
- `weather_020398.py` (Folder 1)
- `weather_6c1670.py` (Folder 1)
- `weather_ac8565.py` (Folder 1)
- `web_779fc7.py` (Folder 1)
- `web_protocol_b74f41.py` (Folder 1)
- `xml_da8497.py` (Folder 1)
- `__init___642017.py` (Folder 2)
- `__init___81f696.py` (Folder 2)
- `__init___b0eb15.py` (Folder 2)
- `__init___e1774c.py` (Folder 2)
- `__main___793293.py` (Folder 2)
- `_main_43a52e.py` (Folder 2)
- `_settings_351566.py` (Folder 2)
- `accessor_9a03f0.py` (Folder 2)
- `accessories_b90bda.py` (Folder 2)
- `agent_c9999e.py` (Folder 2)
- `api_1827f5.py` (Folder 2)
- `app_986ad6.py` (Folder 2)
- `appengine_f79cd5.py` (Folder 2)
- `audit_992d57.py` (Folder 2)
- `backtesting_a4b109.py` (Folder 2)
- `base_41a673.py` (Folder 2)
- `beam_search_54db1c.py` (Folder 2)
- `binary_sensor_8bad86.py` (Folder 2)
- `blocks_219fd5.py` (Folder 2)
- `brendel_bethge_016eab.py` (Folder 2)
- `browse_media_2e1817.py` (Folder 2)
- `client_714ccb.py` (Folder 2)
- `client_ws_7f7caf.py` (Folder 2)
- `climate_5f3cc6.py` (Folder 2)
- `climate_9e8de7.py` (Folder 2)
- `climate_d27245.py` (Folder 2)
- `common_d0ad99.py` (Folder 2)
- `concat_31a224.py` (Folder 2)
- `conftest_04471e.py` (Folder 2)
- `connector_bbd6dd.py` (Folder 2)
- `core_567450.py` (Folder 2)
- `data_b998b2.py` (Folder 2)
- `datetimes_71dd39.py` (Folder 2)
- `differentialevolution_f6d97d.py` (Folder 2)
- `events_b9f3ac.py` (Folder 2)
- `excel_26f18f.py` (Folder 2)
- `expected_catalog_04b2bc.py` (Folder 2)
- `experiments_a52bc9.py` (Folder 2)
- `fields_638399.py` (Folder 2)
- `format_0c602c.py` (Folder 2)
- `format_84da8c.py` (Folder 2)
- `functionlib_51de18.py` (Folder 2)
- `generic_8a9e7c.py` (Folder 2)
- `generic_a07530.py` (Folder 2)
- `git_90c119.py` (Folder 2)
- `groupby_4ba682.py` (Folder 2)
- `gymexperiments_f3782d.py` (Folder 2)
- `handlers_890c61.py` (Folder 2)
- `handlers_facbc6.py` (Folder 2)
- `headers_8374b8.py` (Folder 2)
- `helpers_5f1f91.py` (Folder 2)
- `hist_13f59f.py` (Folder 2)
- `history_utils_a432d7.py` (Folder 2)
- `html_2728b0.py` (Folder 2)
- `http_b073d4.py` (Folder 2)
- `http_models_d4e52a.py` (Folder 2)
- `http_sessions_1ab9ee.py` (Folder 2)
- `httputil_afd03a.py` (Folder 2)
- `humidifier_29087a.py` (Folder 2)
- `legacy_583497.py` (Folder 2)
- `line_ranges_basic_766e5e.py` (Folder 2)
- `main_c8ac67.py` (Folder 2)
- `merge_f40757.py` (Folder 2)
- `message_7557cb.py` (Folder 2)
- `message_cache_dbda93.py` (Folder 2)
- `message_send_b7048c.py` (Folder 2)
- `messaging_30ade3.py` (Folder 2)
- `model_test_case_905237.py` (Folder 2)
- `models_25879e.py` (Folder 2)
- `modern_5fa600.py` (Folder 2)
- `multitask_data_loader_4fc11f.py` (Folder 2)
- `mypy_a965ef.py` (Folder 2)
- `nlp_fe0862.py` (Folder 2)
- `notify_74dbdd.py` (Folder 2)
- `numpy_f7c5e7.py` (Folder 2)
- `oneshot_b70888.py` (Folder 2)
- `optimizers_03ae81.py` (Folder 2)
- `pathfinding_7a5b8f.py` (Folder 2)
- `pivot_1f30db.py` (Folder 2)
- `prefix_7be09c.py` (Folder 2)
- `python_f218d0.py` (Folder 2)
- `query_cache_manager_f69be4.py` (Folder 2)
- `raiden_service_70320b.py` (Folder 2)
- `requests_html_b5031b.py` (Folder 2)
- `response_365c37.py` (Folder 2)
- `retry_71dc8b.py` (Folder 2)
- `rolling_dcc27f.py` (Folder 2)
- `send_email_8ce2b6.py` (Folder 2)
- `sensor_594ee1.py` (Folder 2)
- `sensor_c9d852.py` (Folder 2)
- `simple_influence_296df3.py` (Folder 2)
- `sources_dfba5e.py` (Folder 2)
- `sql_07b300.py` (Folder 2)
- `streams_3625c4.py` (Folder 2)
- `streams_748e41.py` (Folder 2)
- `streams_a2a52b.py` (Folder 2)
- `stripe_8fa3cb.py` (Folder 2)
- `switch_a354e7.py` (Folder 2)
- `test_case_e8713f.py` (Folder 2)
- `test_code_d994fc.py` (Folder 2)
- `test_core_08030e.py` (Folder 2)
- `test_datahandler_39fafd.py` (Folder 2)
- `test_freqtradebot_64c23d.py` (Folder 2)
- `test_rewards_83728f.py` (Folder 2)
- `test_strategy_state_127381.py` (Folder 2)
- `test_url_1ceabd.py` (Folder 2)
- `test_utils_46addc.py` (Folder 2)
- `token_class_8d102b.py` (Folder 2)
- `train_17392c.py` (Folder 2)
- `transformer_module_941d3c.py` (Folder 2)
- `tuples_43cb14.py` (Folder 2)
- `type_adapter_c4334d.py` (Folder 2)
- `unit_system_a3b7c4.py` (Folder 2)
- `user_fbdb95.py` (Folder 2)
- `vocab_563bd2.py` (Folder 2)
- `vocabulary_89f8f0.py` (Folder 2)
- `wandb_4a4375.py` (Folder 2)
- `web_request_930455.py` (Folder 2)
- `worker_083662.py` (Folder 2)
- `xml_bf31a3.py` (Folder 2)
- `__init___716b23.py` (Folder 3)
- `__init___733adb.py` (Folder 3)
- `__init___7c4947.py` (Folder 3)
- `__init___caf114.py` (Folder 3)
- `_parser_ad554d.py` (Folder 3)
- `alarm_control_panel_85f407.py` (Folder 3)
- `api_1f14a8.py` (Folder 3)
- `api_cbdcbd.py` (Folder 3)
- `array_be8945.py` (Folder 3)
- `automations_d10ae8.py` (Folder 3)
- `autopep8_8c1ae5.py` (Folder 3)
- `base_a7536d.py` (Folder 3)
- `base_ef8405.py` (Folder 3)
- `BaseReinforcementLearningModel_998089.py` (Folder 3)
- `beam_search_b3623d.py` (Folder 3)
- `bias_mitigators_adad59.py` (Folder 3)
- `binary_sensor_9e7fb5.py` (Folder 3)
- `binary_sensor_b660c7.py` (Folder 3)
- `binary_sensor_c1e39b.py` (Folder 3)
- `binary_sensor_fae711.py` (Folder 3)
- `blockchain_37e43c.py` (Folder 3)
- `cached_dataset_374c5d.py` (Folder 3)
- `category_59cae5.py` (Folder 3)
- `client_22bde8.py` (Folder 3)
- `client_391ffa.py` (Folder 3)
- `client_exceptions_dac1d7.py` (Folder 3)
- `codegen_d06b21.py` (Folder 3)
- `combined_d7e31f.py` (Folder 3)
- `commands_33f6b3.py` (Folder 3)
- `config_flow_b3a401.py` (Folder 3)
- `configurations_98b5ec.py` (Folder 3)
- `conftest_ddc29c.py` (Folder 3)
- `conftest_f547b4.py` (Folder 3)
- `converter_d3febb.py` (Folder 3)
- `cover_1e7e0c.py` (Folder 3)
- `cyclist_8fa926.py` (Folder 3)
- `databricks_f5496f.py` (Folder 3)
- `device_e2c8fa.py` (Folder 3)
- `device_tracker_406c5a.py` (Folder 3)
- `encoder_8f05c7.py` (Folder 3)
- `entity_deba6b.py` (Folder 3)
- `es_85f924.py` (Folder 3)
- `exceptions_725799.py` (Folder 3)
- `exchange_02f52e.py` (Folder 3)
- `fix_operator_2f0209.py` (Folder 3)
- `floats_47251e.py` (Folder 3)
- `follower_4727fb.py` (Folder 3)
- `freqai_test_strat_0dc220.py` (Folder 3)
- `gallery_2bb1c1.py` (Folder 3)
- `gated_cnn_encoder_38b825.py` (Folder 3)
- `gen_afcffc.py` (Folder 3)
- `generic_test_builders_3da473.py` (Folder 3)
- `geo_location_c45ba4.py` (Folder 3)
- `git_e2f7c0.py` (Folder 3)
- `groupby_7dd4b2.py` (Folder 3)
- `hashing_3bf140.py` (Folder 3)
- `hyperliquid_da9881.py` (Folder 3)
- `image_processing_dff0c0.py` (Folder 3)
- `informative_decorator_cfc496.py` (Folder 3)
- `interface_7aa819.py` (Folder 3)
- `light_1f4c8c.py` (Folder 3)
- `macros_10e8dd.py` (Folder 3)
- `main_be1d12.py` (Folder 3)
- `media_player_8b296d.py` (Folder 3)
- `media_player_b4e2f8.py` (Folder 3)
- `media_player_f3d659.py` (Folder 3)
- `memory_df2954.py` (Folder 3)
- `netutil_1c2f9b.py` (Folder 3)
- `numeric_b9721d.py` (Folder 3)
- `pd_extractors_601e06.py` (Folder 3)
- `pep8_82e145.py` (Folder 3)
- `printing_6fb550.py` (Folder 3)
- `python_lint_handler_d8f325.py` (Folder 3)
- `query_components_beb202.py` (Folder 3)
- `query_context_ca911d.py` (Folder 3)
- `range_eccc67.py` (Folder 3)
- `rate_limiter_5a146f.py` (Folder 3)
- `remote_60424b.py` (Folder 3)
- `remote_ip_c6c5d8.py` (Folder 3)
- `remoteclient_c4932f.py` (Folder 3)
- `reporter_4f7bfa.py` (Folder 3)
- `select_c9b799.py` (Folder 3)
- `sensor_152d62.py` (Folder 3)
- `sensor_1533dd.py` (Folder 3)
- `sensor_440acd.py` (Folder 3)
- `sensor_7457a0.py` (Folder 3)
- `sensor_a11f7f.py` (Folder 3)
- `sensor_ab37f6.py` (Folder 3)
- `sensor_afd46b.py` (Folder 3)
- `sensor_f532d5.py` (Folder 3)
- `sensor_f9ca2f.py` (Folder 3)
- `setup_25159f.py` (Folder 3)
- `show_0b8fd6.py` (Folder 3)
- `simple_httpclient_5928fa.py` (Folder 3)
- `states_8ed126.py` (Folder 3)
- `switch_fa31bd.py` (Folder 3)
- `test_api_6f44ac.py` (Folder 3)
- `test_awsclient_21d219.py` (Folder 3)
- `test_beam_search_c853ca.py` (Folder 3)
- `test_cat_c39959.py` (Folder 3)
- `test_drop_537b51.py` (Folder 3)
- `test_dst_22fce9.py` (Folder 3)
- `test_eval_6a919b.py` (Folder 3)
- `test_feather_09566a.py` (Folder 3)
- `test_first_valid_index_e77512.py` (Folder 3)
- `test_interval_aba39a.py` (Folder 3)
- `test_manifest_b9e428.py` (Folder 3)
- `test_merge_d90586.py` (Folder 3)
- `test_nanops_cc7886.py` (Folder 3)
- `test_operations_088d14.py` (Folder 3)
- `test_pipeline_0fd84d.py` (Folder 3)
- `test_sliceaware_classifier_1e31e5.py` (Folder 3)
- `test_style_48cf89.py` (Folder 3)
- `test_to_csv_e7f2e8.py` (Folder 3)
- `tools_6a9569.py` (Folder 3)
- `transforms_f38536.py` (Folder 3)
- `type_switches_069af3.py` (Folder 3)
- `update_coordinator_155ec7.py` (Folder 3)
- `usage_logger_0d0a99.py` (Folder 3)
- `util_d92a2e.py` (Folder 3)
- `utils_f25548.py` (Folder 3)
- `vacuum_0ba5e2.py` (Folder 3)
- `vacuum_d388a1.py` (Folder 3)
- `voluntary_exits_96290c.py` (Folder 3)
- `web_app_e61508.py` (Folder 3)
- `worker_c5c26f.py` (Folder 3)
- `__init___000e4c.py` (Folder 4)
- `__init___cd0888.py` (Folder 4)
- `_funcs_29fad7.py` (Folder 4)
- `_model_construction_47e7ac.py` (Folder 4)
- `abc_4434cf.py` (Folder 4)
- `actor_865029.py` (Folder 4)
- `additive_noise_84a712.py` (Folder 4)
- `alarm_control_panel_0476ed.py` (Folder 4)
- `alarm_control_panel_7f2de8.py` (Folder 4)
- `base_55e6eb.py` (Folder 4)
- `base_6a0ea7.py` (Folder 4)
- `base_dfd9c5.py` (Folder 4)
- `binary_sensor_9e59d0.py` (Folder 4)
- `binary_sensor_e70512.py` (Folder 4)
- `binary_sensor_f7efe0.py` (Folder 4)
- `cache_f3079a.py` (Folder 4)
- `camera_4798a7.py` (Folder 4)
- `clients_e6ffbd.py` (Folder 4)
- `climate_9505f4.py` (Folder 4)
- `conftest_0be5a6.py` (Folder 4)
- `conftest_3e1e08.py` (Folder 4)
- `conftest_7752c7.py` (Folder 4)
- `conftest_823740.py` (Folder 4)
- `conftest_d0fb86.py` (Folder 4)
- `core_06066e.py` (Folder 4)
- `core_0bbc8d.py` (Folder 4)
- `core_fcc14c.py` (Folder 4)
- `custom_profile_fields_ccef17.py` (Folder 4)
- `data_9f84cd.py` (Folder 4)
- `database_2ff475.py` (Folder 4)
- `device_9b2e80.py` (Folder 4)
- `entity_3c2299.py` (Folder 4)
- `entity_680502.py` (Folder 4)
- `entity_7a1eb0.py` (Folder 4)
- `entity_8bb532.py` (Folder 4)
- `evaluator_beeb92.py` (Folder 4)
- `fenced_code_ee606d.py` (Folder 4)
- `filters_ba549f.py` (Folder 4)
- `find_learning_rate_f33819.py` (Folder 4)
- `geo_location_9da493.py` (Folder 4)
- `hass_imports_8c6b50.py` (Folder 4)
- `html_1d1577.py` (Folder 4)
- `indexing_d02a25.py` (Folder 4)
- `input_reduction_cfaaeb.py` (Folder 4)
- `legacy_8258c7.py` (Folder 4)
- `light_65a387.py` (Folder 4)
- `logger_utils_ef5455.py` (Folder 4)
- `management_f4942e.py` (Folder 4)
- `media_player_d56fd8.py` (Folder 4)
- `melt_24cac4.py` (Folder 4)
- `micropkg_d259a6.py` (Folder 4)
- `misc_7d75ac.py` (Folder 4)
- `missing_a60b28.py` (Folder 4)
- `modular_pipeline_76bcb9.py` (Folder 4)
- `moving_average_64416a.py` (Folder 4)
- `mypy_97629e.py` (Folder 4)
- `narrow_5a6ea2.py` (Folder 4)
- `notify_7c7ee7.py` (Folder 4)
- `outgoing_webhook_e191ed.py` (Folder 4)
- `pipeline_25a0ac.py` (Folder 4)
- `plotting_800aba.py` (Folder 4)
- `plotting_b73057.py` (Folder 4)
- `pytables_63fc81.py` (Folder 4)
- `records_1f490a.py` (Folder 4)
- `refactor_3077ca.py` (Folder 4)
- `remote_939ae3.py` (Folder 4)
- `runnable_108c88.py` (Folder 4)
- `schemas_156b3c.py` (Folder 4)
- `schemas_1abb49.py` (Folder 4)
- `select_d2495d.py` (Folder 4)
- `sensor_010a2f.py` (Folder 4)
- `sensor_97b467.py` (Folder 4)
- `sensor_f004b2.py` (Folder 4)
- `shrinker_03130f.py` (Folder 4)
- `sourcemaps_0942d0.py` (Folder 4)
- `starters_942b24.py` (Folder 4)
- `strategies_b10d9c.py` (Folder 4)
- `streams_0b7f1d.py` (Folder 4)
- `sum__a10910.py` (Folder 4)
- `switch_9b7d9a.py` (Folder 4)
- `test_active_update_coordinator_3372d9.py` (Folder 4)
- `test_arrow_ebc6ab.py` (Folder 4)
- `test_collections_4d5e21.py` (Folder 4)
- `test_config_d874f5.py` (Folder 4)
- `test_deprecation_a6e8da.py` (Folder 4)
- `test_drop_duplicates_d918e0.py` (Folder 4)
- `test_dtypes_1ace40.py` (Folder 4)
- `test_info_8bfc42.py` (Folder 4)
- `test_interval_e04786.py` (Folder 4)
- `test_kraken_f9036a.py` (Folder 4)
- `test_layers_95be9f.py` (Folder 4)
- `test_loc_3f3aec.py` (Folder 4)
- `test_model_signature_86e541.py` (Folder 4)
- `test_patching_0f4c9f.py` (Folder 4)
- `test_pipeline_ab4ea1.py` (Folder 4)
- `test_process_custody_slashing_e5de54.py` (Folder 4)
- `test_process_execution_payload_bd0176.py` (Folder 4)
- `test_project_1f989e.py` (Folder 4)
- `test_reductions_ebf449.py` (Folder 4)
- `test_registry_31caeb.py` (Folder 4)
- `test_run_operations_c6edf5.py` (Folder 4)
- `test_sample_mode_75a9c6.py` (Folder 4)
- `test_selectors_267ea0.py` (Folder 4)
- `test_stack_b5f4bd.py` (Folder 4)
- `test_tags_ccc8b0.py` (Folder 4)
- `test_targetstate_85ad01.py` (Folder 4)
- `test_truncate_a84f32.py` (Folder 4)
- `test_util_8bae1f.py` (Folder 4)
- `text_classification_json_d0a412.py` (Folder 4)
- `timedeltas_44fa81.py` (Folder 4)
- `trade_converter_316324.py` (Folder 4)
- `transactions_dad89f.py` (Folder 4)
- `transformer_layer_bf9c8e.py` (Folder 4)
- `typing_c8a53e.py` (Folder 4)
- `v0_fbd680.py` (Folder 4)
- `variables_c92b2f.py` (Folder 4)
- `web_routedef_a1a149.py` (Folder 4)
- `worker_1df819.py` (Folder 4)
- `yh_clienttrader_6b6d1f.py` (Folder 4)
- `__init___1a62a7.py` (Folder 5)
- `__init___943365.py` (Folder 5)
- `__init___d9ab30.py` (Folder 5)
- `__init___ea74db.py` (Folder 5)
- `alert_tests_e8cf0b.py` (Folder 5)
- `alerta_gitlab_b4eabf.py` (Folder 5)
- `alexa_config_96590b.py` (Folder 5)
- `api_fb33d1.py` (Folder 5)
- `apply_575d8e.py` (Folder 5)
- `async_queries_tests_ab1690.py` (Folder 5)
- `base_44bf21.py` (Folder 5)
- `base_api_2a8768.py` (Folder 5)
- `basic_example_9b8645.py` (Folder 5)
- `binary_sensor_66e712.py` (Folder 5)
- `binary_sensor_88cfce.py` (Folder 5)
- `binary_sensor_f0d918.py` (Folder 5)
- `browse_media_f0c649.py` (Folder 5)
- `browser_8130a6.py` (Folder 5)
- `camera_19b416.py` (Folder 5)
- `camera_4dd787.py` (Folder 5)
- `cancel_github_workflows_0b6cfa.py` (Folder 5)
- `choice_599488.py` (Folder 5)
- `class_validators_8bdfa8.py` (Folder 5)
- `color_a02da4.py` (Folder 5)
- `common_285794.py` (Folder 5)
- `compiler38_61d171.py` (Folder 5)
- `compiler_3b1181.py` (Folder 5)
- `complete_func_args_83358f.py` (Folder 5)
- `conditional_random_field_4f2ecd.py` (Folder 5)
- `conftest_bc6db5.py` (Folder 5)
- `conftest_ca8ed3.py` (Folder 5)
- `converter_489a6d.py` (Folder 5)
- `coordinator_a3c28d.py` (Folder 5)
- `core_711830.py` (Folder 5)
- `core_9c7c0d.py` (Folder 5)
- `datetime_helpers_3d4ed3.py` (Folder 5)
- `decoder_5a1e74.py` (Folder 5)
- `discovery_3110d1.py` (Folder 5)
- `entities_19afaa.py` (Folder 5)
- `entity_27a2b6.py` (Folder 5)
- `entity_428030.py` (Folder 5)
- `entity_543df3.py` (Folder 5)
- `entity_5876c1.py` (Folder 5)
- `entity_9f065b.py` (Folder 5)
- `entity_e12cd0.py` (Folder 5)
- `entity_f58cea.py` (Folder 5)
- `errors_7ed27c.py` (Folder 5)
- `faustdocs_7d8e30.py` (Folder 5)
- `featureflags_544034.py` (Folder 5)
- `file_name_dac33e.py` (Folder 5)
- `forwarder_298c02.py` (Folder 5)
- `functions_b12404.py` (Folder 5)
- `gas_reserve_f132a8.py` (Folder 5)
- `info_58f159.py` (Folder 5)
- `instrumentations_examples_d8ae97.py` (Folder 5)
- `iresolver_51e6b1.py` (Folder 5)
- `learning_rate_scheduler_b929ac.py` (Folder 5)
- `light_7b6626.py` (Folder 5)
- `line_ranges_exceeding_end_de8e88.py` (Folder 5)
- `lock_c9b41d.py` (Folder 5)
- `locking_996032.py` (Folder 5)
- `masked_cd87cc.py` (Folder 5)
- `media_player_7d6a6e.py` (Folder 5)
- `metamodel_f9ba03.py` (Folder 5)
- `metastore_cache_b6b4a1.py` (Folder 5)
- `metrics_195f9f.py` (Folder 5)
- `mi_fgsm_09ba33.py` (Folder 5)
- `microbatch_743d2f.py` (Folder 5)
- `mysql_113041.py` (Folder 5)
- `nodeiterator_94d3a3.py` (Folder 5)
- `number_537226.py` (Folder 5)
- `openapi_a5fc11.py` (Folder 5)
- `period_0be655.py` (Folder 5)
- `player_035866.py` (Folder 5)
- `pool_9fd38f.py` (Folder 5)
- `presto_08b2cb.py` (Folder 5)
- `providers_d61054.py` (Folder 5)
- `recastlib_49d2ed.py` (Folder 5)
- `redis_utils_abe297.py` (Folder 5)
- `rolling_129721.py` (Folder 5)
- `scheduler_tests_223cad.py` (Folder 5)
- `schema_8a7298.py` (Folder 5)
- `secret_registry_13acaf.py` (Folder 5)
- `sensor_132dfc.py` (Folder 5)
- `sensor_20a298.py` (Folder 5)
- `sensor_457aac.py` (Folder 5)
- `sensor_a7ec9e.py` (Folder 5)
- `sensor_b70d9a.py` (Folder 5)
- `sensor_cdccc8.py` (Folder 5)
- `sensor_dd9269.py` (Folder 5)
- `sensor_f62020.py` (Folder 5)
- `serializers_307391.py` (Folder 5)
- `server_80c298.py` (Folder 5)
- `service_registry_e94e0b.py` (Folder 5)
- `sets_3f50ca.py` (Folder 5)
- `simple_data_loader_a4f8f3.py` (Folder 5)
- `span_based_f1_measure_cf6fc9.py` (Folder 5)
- `startup_7ab634.py` (Folder 5)
- `subscription_info_2fc87f.py` (Folder 5)
- `test_compression_088712.py` (Folder 5)
- `test_compression_21f3e4.py` (Folder 5)
- `test_cumulative_8df6e0.py` (Folder 5)
- `test_decorators_98d50d.py` (Folder 5)
- `test_encoding_e05f99.py` (Folder 5)
- `test_features_a57fa5.py` (Folder 5)
- `test_finality_a16bde.py` (Folder 5)
- `test_indexing_3e5a66.py` (Folder 5)
- `test_local_0cfc7f.py` (Folder 5)
- `test_masked_a94e75.py` (Folder 5)
- `test_message_object_cf66b4.py` (Folder 5)
- `test_previous_version_state_eb5537.py` (Folder 5)
- `test_procedure_3f923d.py` (Folder 5)
- `test_process_justification_and_finalization_fc33da.py` (Folder 5)
- `test_proxy_functional_150ce7.py` (Folder 5)
- `test_tz_localize_a90669.py` (Folder 5)
- `test_tz_localize_ae2d56.py` (Folder 5)
- `test_week_31892f.py` (Folder 5)
- `test_writers_3a0994.py` (Folder 5)
- `tls_connection_607baf.py` (Folder 5)
- `transformer_embeddings_test_676185.py` (Folder 5)
- `transport_2d9279.py` (Folder 5)
- `withdrawals_8c07d5.py` (Folder 5)
- `__init___8f577b.py` (Folder 6)
- `__init___9fcc21.py` (Folder 6)
- `__init___c2e1cb.py` (Folder 6)
- `__init___c5e9e5.py` (Folder 6)
- `_dataclasses_645d3b.py` (Folder 6)
- `_normalize_0877ea.py` (Folder 6)
- `_robot_tester_77d42f.py` (Folder 6)
- `_xlsxwriter_8ba0cd.py` (Folder 6)
- `alarm_control_panel_489ba5.py` (Folder 6)
- `base_384787.py` (Folder 6)
- `base_7e78cf.py` (Folder 6)
- `base_7e9ae1.py` (Folder 6)
- `billing_page_c98d2b.py` (Folder 6)
- `binary_sensor_254d71.py` (Folder 6)
- `binary_sensor_25691a.py` (Folder 6)
- `blocks_3a122c.py` (Folder 6)
- `cache_053dd3.py` (Folder 6)
- `cache_helpers_e012c1.py` (Folder 6)
- `calibration_d3fe81.py` (Folder 6)
- `check_config_125c88.py` (Folder 6)
- `climate_8c1cee.py` (Folder 6)
- `common_dde807.py` (Folder 6)
- `common_fcbb68.py` (Folder 6)
- `conftest_0f5d02.py` (Folder 6)
- `conftest_81d476.py` (Folder 6)
- `conftest_cc5a31.py` (Folder 6)
- `core_880b1d.py` (Folder 6)
- `cover_b92bd5.py` (Folder 6)
- `data_collator_b6829a.py` (Folder 6)
- `data_drawer_b3adf2.py` (Folder 6)
- `device_e3bccf.py` (Folder 6)
- `document_48bf0e.py` (Folder 6)
- `ecs_worker_ea5d21.py` (Folder 6)
- `encoding_93b3e1.py` (Folder 6)
- `entity_297566.py` (Folder 6)
- `entity_a0478b.py` (Folder 6)
- `entity_ad6171.py` (Folder 6)
- `entity_c33184.py` (Folder 6)
- `entityfilter_421d82.py` (Folder 6)
- `errors_174875.py` (Folder 6)
- `evalb_bracketing_scorer_test_a77d39.py` (Folder 6)
- `exceptions_da9151.py` (Folder 6)
- `exchange_ws_9ca098.py` (Folder 6)
- `funcdef_return_type_trailing_comma_6ef842.py` (Folder 6)
- `ghostwriter_f75a80.py` (Folder 6)
- `groupby_241e8f.py` (Folder 6)
- `image_processing_f9093d.py` (Folder 6)
- `impl_a6397c.py` (Folder 6)
- `internal_244784.py` (Folder 6)
- `invites_f24478.py` (Folder 6)
- `json_5bd122.py` (Folder 6)
- `json_utils_d4639e.py` (Folder 6)
- `kraken_c1d78e.py` (Folder 6)
- `light_638354.py` (Folder 6)
- `loggers_97a372.py` (Folder 6)
- `media_browser_2b5d62.py` (Folder 6)
- `media_player_4a0667.py` (Folder 6)
- `media_player_b4a5f5.py` (Folder 6)
- `messages_0d6d46.py` (Folder 6)
- `notify_b3032a.py` (Folder 6)
- `parse_7b93d1.py` (Folder 6)
- `processor_23c7a3.py` (Folder 6)
- `processutils_97fd75.py` (Folder 6)
- `project_2d0af1.py` (Folder 6)
- `providers_fc28a2.py` (Folder 6)
- `pytree_2ac25d.py` (Folder 6)
- `recorder_29bf33.py` (Folder 6)
- `responses_7e84ca.py` (Folder 6)
- `rouge_test_960688.py` (Folder 6)
- `rpc_0d6cda.py` (Folder 6)
- `run_f6671c.py` (Folder 6)
- `select_ce6a0c.py` (Folder 6)
- `sensor_279169.py` (Folder 6)
- `sensor_31ba0c.py` (Folder 6)
- `sensor_48e774.py` (Folder 6)
- `sensor_6a8a98.py` (Folder 6)
- `sensor_6b9561.py` (Folder 6)
- `sensor_912d78.py` (Folder 6)
- `sensor_e4b539.py` (Folder 6)
- `sensor_fdf575.py` (Folder 6)
- `shard_block_395422.py` (Folder 6)
- `staticfiles_164434.py` (Folder 6)
- `storage_ecdafa.py` (Folder 6)
- `switch_5119b1.py` (Folder 6)
- `switch_5474d5.py` (Folder 6)
- `switch_8d0b59.py` (Folder 6)
- `switch_aac10e.py` (Folder 6)
- `switch_e3332d.py` (Folder 6)
- `sync_committee_b34f1b.py` (Folder 6)
- `syrupy_b69260.py` (Folder 6)
- `target_8f735b.py` (Folder 6)
- `tcpserver_3f34d3.py` (Folder 6)
- `template_0e4e35.py` (Folder 6)
- `test_backup_restore_171613.py` (Folder 6)
- `test_bybit_c7e0f9.py` (Folder 6)
- `test_converter_703ec2.py` (Folder 6)
- `test_deprecated_validate_arguments_bca830.py` (Folder 6)
- `test_incremental_merge_exclude_columns_c80032.py` (Folder 6)
- `test_mediation_fee_82ecfe.py` (Folder 6)
- `test_profile_dir_8b1317.py` (Folder 6)
- `test_rest_client_6dfb62.py` (Folder 6)
- `test_spark_0e0bdf.py` (Folder 6)
- `test_websocket_handshake_a4bf54.py` (Folder 6)
- `tile_a06764.py` (Folder 6)
- `token_49945a.py` (Folder 6)
- `transactions_7edb11.py` (Folder 6)
- `translations_f1af2c.py` (Folder 6)
- `transports_a66523.py` (Folder 6)
- `tts_37e6e2.py` (Folder 6)
- `utils_0e78a9.py` (Folder 6)
- `utils_5006d0.py` (Folder 6)
- `utils_b5d4df.py` (Folder 6)
- `utils_eb1e68.py` (Folder 6)
- `vacuum_4633b7.py` (Folder 6)
- `valve_d875ca.py` (Folder 6)
- `web_6fce4a.py` (Folder 6)
- `xq_follower_0c252a.py` (Folder 6)
- `__init___49b7e6.py` (Folder 7)
- `__init___605457.py` (Folder 7)
- `__init___97fa7b.py` (Folder 7)
- `__init___a5a747.py` (Folder 7)
- `_cert_chain_analyzer_dafae6.py` (Folder 7)
- `_schema_validator_f2fb4b.py` (Folder 7)
- `_warnings_0608d5.py` (Folder 7)
- `alerta_4aa3fa.py` (Folder 7)
- `avatar_3ae540.py` (Folder 7)
- `base_165d8d.py` (Folder 7)
- `base_34b940.py` (Folder 7)
- `base_9a2f6b.py` (Folder 7)
- `base_ac17ae.py` (Folder 7)
- `base_d18b00.py` (Folder 7)
- `base_parser_95aa6d.py` (Folder 7)
- `bases_be73fc.py` (Folder 7)
- `bimpm_matching_1894c7.py` (Folder 7)
- `binary_sensor_19876e.py` (Folder 7)
- `bucket_batch_sampler_a1e388.py` (Folder 7)
- `categorical_2bf83a.py` (Folder 7)
- `checkpointer_2adffa.py` (Folder 7)
- `climate_e3517e.py` (Folder 7)
- `common_888838.py` (Folder 7)
- `compilation_283ca2.py` (Folder 7)
- `conf_6893b0.py` (Folder 7)
- `conftest_4ecb9c.py` (Folder 7)
- `conftest_db6ccc.py` (Folder 7)
- `container_instance_af6637.py` (Folder 7)
- `cookiejar_1ce0bd.py` (Folder 7)
- `coordinator_06fdbe.py` (Folder 7)
- `coordinator_bd80ae.py` (Folder 7)
- `counts_1f7027.py` (Folder 7)
- `dependencies_8f19df.py` (Folder 7)
- `discovery_flow_f269c4.py` (Folder 7)
- `driver_354fe6.py` (Folder 7)
- `encoder_1dd86d.py` (Folder 7)
- `entropy_test_fa6663.py` (Folder 7)
- `externalbo_9ead38.py` (Folder 7)
- `filesystem_970993.py` (Folder 7)
- `fmtonoff_3a4c86.py` (Folder 7)
- `fork_transition_eab473.py` (Folder 7)
- `formdata_aebdbf.py` (Folder 7)
- `frame_manager_271dac.py` (Folder 7)
- `functional_serializers_f18005.py` (Folder 7)
- `gen_3e3f5c.py` (Folder 7)
- `http_writer_4daf2a.py` (Folder 7)
- `IPairList_2e2b02.py` (Folder 7)
- `key_d8a5a1.py` (Folder 7)
- `kill_if_no_output_54c848.py` (Folder 7)
- `lazy_a1ea35.py` (Folder 7)
- `log_fbbf94.py` (Folder 7)
- `log_manager_0808f7.py` (Folder 7)
- `main_6bc0d1.py` (Folder 7)
- `manager_test_d62131.py` (Folder 7)
- `masked_reductions_91c9b6.py` (Folder 7)
- `minserver_4f13e8.py` (Folder 7)
- `models_234e77.py` (Folder 7)
- `models_e47d98.py` (Folder 7)
- `node_4a9aca.py` (Folder 7)
- `oauth2_4b34aa.py` (Folder 7)
- `omegaconf_config_2d1798.py` (Folder 7)
- `openapi_af3f12.py` (Folder 7)
- `ops_a279aa.py` (Folder 7)
- `parse_ae4d25.py` (Folder 7)
- `pin_fe5cb3.py` (Folder 7)
- `pow_5b6282.py` (Folder 7)
- `random_value_d2191f.py` (Folder 7)
- `ref_resolver_4b8223.py` (Folder 7)
- `region_detector_fc1ccc.py` (Folder 7)
- `regions_ce690d.py` (Folder 7)
- `run_input_54b93c.py` (Folder 7)
- `runtime_8a2834.py` (Folder 7)
- `sasreader_57b625.py` (Folder 7)
- `schema_663c54.py` (Folder 7)
- `schema_ee7352.py` (Folder 7)
- `sensor_5c46f4.py` (Folder 7)
- `sensor_677086.py` (Folder 7)
- `sensor_f749b9.py` (Folder 7)
- `sorting_5bda3c.py` (Folder 7)
- `state_report_ea72b9.py` (Folder 7)
- `statistics_05be06.py` (Folder 7)
- `stats_logger_411bed.py` (Folder 7)
- `strings_c18fe4.py` (Folder 7)
- `switch_045d67.py` (Folder 7)
- `switch_959c09.py` (Folder 7)
- `switch_abf1fa.py` (Folder 7)
- `switch_ebb0e1.py` (Folder 7)
- `table_e760ca.py` (Folder 7)
- `template_parser_ecf506.py` (Folder 7)
- `test_api_917878.py` (Folder 7)
- `test_app_9a2346.py` (Folder 7)
- `test_assert_frame_equal_d9bb56.py` (Folder 7)
- `test_auth_8863c0.py` (Folder 7)
- `test_base_7e51af.py` (Folder 7)
- `test_black_2d705a.py` (Folder 7)
- `test_discovery_ability_c49e14.py` (Folder 7)
- `test_freqai_datakitchen_145f2f.py` (Folder 7)
- `test_get_accessories_34824a.py` (Folder 7)
- `test_incremental_unique_id_a93cab.py` (Folder 7)
- `test_mediatedtransfer_invalid_c11287.py` (Folder 7)
- `test_memory_dataset_822706.py` (Folder 7)
- `test_regression_05a4bf.py` (Folder 7)
- `test_remotepairlist_e175cd.py` (Folder 7)
- `test_session_hook_manager_b0d853.py` (Folder 7)
- `test_setops_dfaf2d.py` (Folder 7)
- `test_stubs_2856c8.py` (Folder 7)
- `tts_59d132.py` (Folder 7)
- `types_263491.py` (Folder 7)
- `vacuum_0d02fa.py` (Folder 7)
- `wrap_modes_1cd0d1.py` (Folder 7)
- `__init___49f91b.py` (Folder 8)
- `__init___8cfb7b.py` (Folder 8)
- `__init___fc1060.py` (Folder 8)
- `_content_aabc8a.py` (Folder 8)
- `_patching_7a8966.py` (Folder 8)
- `_types_6d212e.py` (Folder 8)
- `adjacency_field_20cd46.py` (Folder 8)
- `analysis_98a850.py` (Folder 8)
- `api_88b699.py` (Folder 8)
- `base_75712c.py` (Folder 8)
- `base_7f1c49.py` (Folder 8)
- `basic_iterative_method_79cb2a.py` (Folder 8)
- `bias_mitigator_wrappers_4e83be.py` (Folder 8)
- `binary_sensor_0e5f46.py` (Folder 8)
- `binary_sensor_950ee8.py` (Folder 8)
- `binary_sensor_a3bb34.py` (Folder 8)
- `boolean_ca5b18.py` (Folder 8)
- `broken_futures_strategies_211867.py` (Folder 8)
- `browse_media_bb064b.py` (Folder 8)
- `calendar_b5cdc9.py` (Folder 8)
- `camera_f2981c.py` (Folder 8)
- `charmap_8b24d1.py` (Folder 8)
- `cli_d3e92b.py` (Folder 8)
- `cli_e1dc38.py` (Folder 8)
- `client_dc7832.py` (Folder 8)
- `climate_2986cb.py` (Folder 8)
- `common_9cce3e.py` (Folder 8)
- `config_entry_oauth2_flow_4660ef.py` (Folder 8)
- `config_test_89096c.py` (Folder 8)
- `conftest_438921.py` (Folder 8)
- `conftest_71616d.py` (Folder 8)
- `coordinator_144ee7.py` (Folder 8)
- `copartitioned_assignor_d68f8c.py` (Folder 8)
- `cover_b5d82f.py` (Folder 8)
- `cover_c586a7.py` (Folder 8)
- `data_entry_flow_539385.py` (Folder 8)
- `dataset_reader_test_19a9b5.py` (Folder 8)
- `date_parser_e0540e.py` (Folder 8)
- `debias_0a65ea.py` (Folder 8)
- `decimalspace_eba8c8.py` (Folder 8)
- `default_agent_43d164.py` (Folder 8)
- `drill_3ecdbc.py` (Folder 8)
- `entity_015772.py` (Folder 8)
- `entity_14bdff.py` (Folder 8)
- `entity_9725b3.py` (Folder 8)
- `exceptions_667b5e.py` (Folder 8)
- `execution_context_66f661.py` (Folder 8)
- `extract_d1f231.py` (Folder 8)
- `fan_f7fc07.py` (Folder 8)
- `files_a4064b.py` (Folder 8)
- `filters_ef9b58.py` (Folder 8)
- `formatters_73c936.py` (Folder 8)
- `from_params_be1a87.py` (Folder 8)
- `generate_868fef.py` (Folder 8)
- `helpers_cd3297.py` (Folder 8)
- `hls_75e690.py` (Folder 8)
- `image_processing_d971eb.py` (Folder 8)
- `impala_f08d97.py` (Folder 8)
- `jinja_context_427571.py` (Folder 8)
- `light_client_sync_2b8d7f.py` (Folder 8)
- `lock_0a8494.py` (Folder 8)
- `lr_scheduler_b7c77b.py` (Folder 8)
- `manager_a874dc.py` (Folder 8)
- `memory_02aa2d.py` (Folder 8)
- `message_6587fb.py` (Folder 8)
- `numba__bec0d1.py` (Folder 8)
- `number_7ed601.py` (Folder 8)
- `number_9501ee.py` (Folder 8)
- `output_fc837c.py` (Folder 8)
- `parallel_runner_14dbef.py` (Folder 8)
- `parse_84003e.py` (Folder 8)
- `payment_channel_7c5624.py` (Folder 8)
- `photonics_eb3fae.py` (Folder 8)
- `pipeline_a41664.py` (Folder 8)
- `protocol_e4bfc3.py` (Folder 8)
- `pycodestyle_b15d51.py` (Folder 8)
- `rocket_ddf628.py` (Folder 8)
- `runner_1ccfed.py` (Folder 8)
- `schema_basic_5d028f.py` (Folder 8)
- `sensor_82981a.py` (Folder 8)
- `sensor_9ac883.py` (Folder 8)
- `sensor_b3b9e2.py` (Folder 8)
- `sensor_d12c1d.py` (Folder 8)
- `sensor_e842c4.py` (Folder 8)
- `sensor_f8d27b.py` (Folder 8)
- `sensors_be4bf6.py` (Folder 8)
- `sequence_tagging_9c6a51.py` (Folder 8)
- `sequences_74f8b6.py` (Folder 8)
- `smoketest_dda837.py` (Folder 8)
- `storage_1f5683.py` (Folder 8)
- `stream_info_00b540.py` (Folder 8)
- `structures_df39dc.py` (Folder 8)
- `switch_19b62e.py` (Folder 8)
- `switch_7ab363.py` (Folder 8)
- `test_align_09e7a1.py` (Folder 8)
- `test_blocks_31cb5c.py` (Folder 8)
- `test_callbacks_975bd8.py` (Folder 8)
- `test_constructors_111932.py` (Folder 8)
- `test_coop_settle_fd62d3.py` (Folder 8)
- `test_events_5e747c.py` (Folder 8)
- `test_extension_socket_server_334931.py` (Folder 8)
- `test_groupby_dropna_4126e9.py` (Folder 8)
- `test_indexing_f4eafd.py` (Folder 8)
- `test_logical_ops_b270a9.py` (Folder 8)
- `test_micropkg_package_607bd9.py` (Folder 8)
- `test_mssql_35a1b8.py` (Folder 8)
- `test_objects_7b7ddb.py` (Folder 8)
- `test_parsing_6c11a6.py` (Folder 8)
- `test_postgres_3e4674.py` (Folder 8)
- `test_prometheus_2a91bc.py` (Folder 8)
- `test_reductions_f817d7.py` (Folder 8)
- `test_run_app_97664a.py` (Folder 8)
- `test_utils_f58fb9.py` (Folder 8)
- `test_validator_417f9a.py` (Folder 8)
- `test_web_response_099ecd.py` (Folder 8)
- `test_wrappers_53817b.py` (Folder 8)
- `train_0b4066.py` (Folder 8)
- `transactions_e11efe.py` (Folder 8)
- `transports_16d522.py` (Folder 8)
- `typing_bc451b.py` (Folder 8)
- `upnp_cf635d.py` (Folder 8)
- `utils_26b85c.py` (Folder 8)
- `views_b9c7c9.py` (Folder 8)
- `voip_c111eb.py` (Folder 8)
- `xqtrader_85cb19.py` (Folder 8)
- `zulip_tools_3cb6ca.py` (Folder 8)
- `__init___98a815.py` (Folder 9)
- `__init___d06494.py` (Folder 9)
- `__init___d2d86c.py` (Folder 9)
- `__init___e1c908.py` (Folder 9)
- `_config_c8d42c.py` (Folder 9)
- `_version_c1bed8.py` (Folder 9)
- `agents_169617.py` (Folder 9)
- `aiogevent_6d18e7.py` (Folder 9)
- `alarm_control_panel_8a8027.py` (Folder 9)
- `api_tests_8555aa.py` (Folder 9)
- `arguments_9b0c43.py` (Folder 9)
- `binary_info_b9349d.py` (Folder 9)
- `binary_sensor_48c1f5.py` (Folder 9)
- `camera_b201a1.py` (Folder 9)
- `category_fe774f.py` (Folder 9)
- `client_processing_dfe140.py` (Folder 9)
- `common_9fe0bd.py` (Folder 9)
- `computation_233984.py` (Folder 9)
- `conftest_51ed26.py` (Folder 9)
- `conftest_6c0bda.py` (Folder 9)
- `connection_bdafbf.py` (Folder 9)
- `console_aefea3.py` (Folder 9)
- `construct_bbdfda.py` (Folder 9)
- `construction_4e8f89.py` (Folder 9)
- `coordinator_04d05c.py` (Folder 9)
- `core_06cca6.py` (Folder 9)
- `create_596115.py` (Folder 9)
- `datetimes_c6dec9.py` (Folder 9)
- `decorator_5f0cc9.py` (Folder 9)
- `device_tracker_e3d3bd.py` (Folder 9)
- `dict_import_export_tests_cc9da9.py` (Folder 9)
- `discovery_df519d.py` (Folder 9)
- `encrypt_801f53.py` (Folder 9)
- `entity_7014e6.py` (Folder 9)
- `entity_7fe144.py` (Folder 9)
- `entity_95f03c.py` (Folder 9)
- `entity_c0bb5d.py` (Folder 9)
- `entity_f96f8a.py` (Folder 9)
- `eth_node_b6b2aa.py` (Folder 9)
- `event_status_cf4e04.py` (Folder 9)
- `events_5936f7.py` (Folder 9)
- `fast_minimum_norm_f817b7.py` (Folder 9)
- `fixtures_25b262.py` (Folder 9)
- `gen_49c34e.py` (Folder 9)
- `generate_decoder_states_c34463.py` (Folder 9)
- `git_adaf30.py` (Folder 9)
- `humidifier_50d827.py` (Folder 9)
- `idatahandler_cafafa.py` (Folder 9)
- `image_processing_2ba05a.py` (Folder 9)
- `import_realm_55d1ca.py` (Folder 9)
- `interrupt_51464c.py` (Folder 9)
- `ioloop_93815f.py` (Folder 9)
- `light_102551.py` (Folder 9)
- `locale_242de4.py` (Folder 9)
- `make_cf6465.py` (Folder 9)
- `managers_6d6254.py` (Folder 9)
- `media_player_846062.py` (Folder 9)
- `media_player_b7aa19.py` (Folder 9)
- `media_source_23bafc.py` (Folder 9)
- `media_source_e93ead.py` (Folder 9)
- `multigym_1a384d.py` (Folder 9)
- `numba__13f34f.py` (Folder 9)
- `package_1bad75.py` (Folder 9)
- `parquetdatahandler_7da7b1.py` (Folder 9)
- `parser_c026a4.py` (Folder 9)
- `push_notifications_9b9efe.py` (Folder 9)
- `recaster_e3cc77.py` (Folder 9)
- `references_c4be98.py` (Folder 9)
- `regex_202471.py` (Folder 9)
- `schemas_911003.py` (Folder 9)
- `schemas_b7d3bb.py` (Folder 9)
- `sensor_2ab641.py` (Folder 9)
- `sensor_5e7bec.py` (Folder 9)
- `sensor_6d2787.py` (Folder 9)
- `sensor_75a974.py` (Folder 9)
- `sensor_ba4d93.py` (Folder 9)
- `sensor_e99c97.py` (Folder 9)
- `sensor_f43589.py` (Folder 9)
- `server_b7a3d9.py` (Folder 9)
- `setup_9ddcc5.py` (Folder 9)
- `should_validate_4a9594.py` (Folder 9)
- `snowflake_c4fab3.py` (Folder 9)
- `ssl__3d74a8.py` (Folder 9)
- `storage_47c1fc.py` (Folder 9)
- `stream_subscription_0a3612.py` (Folder 9)
- `strings_7e2259.py` (Folder 9)
- `switch_0951c1.py` (Folder 9)
- `switch_79387a.py` (Folder 9)
- `tensorboard_c446d9.py` (Folder 9)
- `test_clickhouse_fb8fef.py` (Folder 9)
- `test_compilation_513485.py` (Folder 9)
- `test_concurrency_sync_792afd.py` (Folder 9)
- `test_constructors_02070a.py` (Folder 9)
- `test_expected_output_03115c.py` (Folder 9)
- `test_fields_216131.py` (Folder 9)
- `test_graph_69fed4.py` (Folder 9)
- `test_groupby_1fa427.py` (Folder 9)
- `test_incremental_predicates_b8a35a.py` (Folder 9)
- `test_indexing_8b3a8e.py` (Folder 9)
- `test_integration_pfs_08c03f.py` (Folder 9)
- `test_legacy_42c41a.py` (Folder 9)
- `test_main_0a06c6.py` (Folder 9)
- `test_merge_ordered_3a00fd.py` (Folder 9)
- `test_meta_learners_15283b.py` (Folder 9)
- `test_moments_consistency_ewm_56b6c2.py` (Folder 9)
- `test_numba_a7df98.py` (Folder 9)
- `test_numeric_6b9cc4.py` (Folder 9)
- `test_numeric_only_37163c.py` (Folder 9)
- `test_package_7fd20e.py` (Folder 9)
- `test_process_block_header_67cc9f.py` (Folder 9)
- `test_process_voluntary_exit_d70212.py` (Folder 9)
- `test_qcut_5ee678.py` (Folder 9)
- `test_raidenservice_d5e26d.py` (Folder 9)
- `test_root_model_d5f7e7.py` (Folder 9)
- `test_selector_29872e.py` (Folder 9)
- `test_sensor_062284.py` (Folder 9)
- `test_setops_3892f2.py` (Folder 9)
- `test_style_33489e.py` (Folder 9)
- `test_webserver_68a1ed.py` (Folder 9)
- `trade_model_19dfec.py` (Folder 9)
- `transformer_889743.py` (Folder 9)
- `tts_72d912.py` (Folder 9)
- `typecheck_tests_125e9e.py` (Folder 9)
- `update_3b91e3.py` (Folder 9)
- `util_b72b9c.py` (Folder 9)
- `v2_validated_func_97b85a.py` (Folder 9)
- `validate_5d15ea.py` (Folder 9)
- `warnings_2ab542.py` (Folder 9)
- `weather_898f49.py` (Folder 9)
- `web_fileresponse_e008e1.py` (Folder 9)
- `websocket_api_4e3223.py` (Folder 9)
- `worker_aa3540.py` (Folder 9)
- `__init___063bfe.py` (Folder 10)
- `__init___38b704.py` (Folder 10)
- `__init___3afdb9.py` (Folder 10)
- `__init___4a14cc.py` (Folder 10)
- `__init___626f0a.py` (Folder 10)
- `__init___c7b03c.py` (Folder 10)
- `_validators_b0a062.py` (Folder 10)
- `account_dbed71.py` (Folder 10)
- `algorithms_faa87b.py` (Folder 10)
- `api_tests_dc31f9.py` (Folder 10)
- `asyncio_606939.py` (Folder 10)
- `benchmark_ocl_3560d9.py` (Folder 10)
- `bias_mitigator_applicator_311808.py` (Folder 10)
- `binary_sensor_c10ab3.py` (Folder 10)
- `binding_d7e1a8.py` (Folder 10)
- `camera_ca0857.py` (Folder 10)
- `categorical_909da8.py` (Folder 10)
- `channel_771f46.py` (Folder 10)
- `class_validators_4df509.py` (Folder 10)
- `climate_1db5cd.py` (Folder 10)
- `common_5f186a.py` (Folder 10)
- `conf_dd0836.py` (Folder 10)
- `conftest_6b2a96.py` (Folder 10)
- `conftest_721c82.py` (Folder 10)
- `core_e8d42b.py` (Folder 10)
- `cover_860401.py` (Folder 10)
- `curl_httpclient_ceb406.py` (Folder 10)
- `data_kitchen_58b770.py` (Folder 10)
- `datetimelike_173f1e.py` (Folder 10)
- `decorator_033664.py` (Folder 10)
- `doris_ca5329.py` (Folder 10)
- `engine_5567f1.py` (Folder 10)
- `entity_8dbc24.py` (Folder 10)
- `eval_431417.py` (Folder 10)
- `expanding_c12a40.py` (Folder 10)
- `fast_gradient_method_5b1a4e.py` (Folder 10)
- `fields_fcc951.py` (Folder 10)
- `ft_rest_client_35ac52.py` (Folder 10)
- `gate_b7f06b.py` (Folder 10)
- `gradient_descent_base_ecdf96.py` (Folder 10)
- `heartbeat_3ab523.py` (Folder 10)
- `http1connection_0621ea.py` (Folder 10)
- `hyperopt_output_c12b12.py` (Folder 10)
- `image_6f55bc.py` (Folder 10)
- `joinquant_follower_7b86fd.py` (Folder 10)
- `json_schema_320197.py` (Folder 10)
- `knx_selector_99cc97.py` (Folder 10)
- `lark_0111f4.py` (Folder 10)
- `light_3c0a4b.py` (Folder 10)
- `light_client_data_collection_12f9fe.py` (Folder 10)
- `local_f7f151.py` (Folder 10)
- `macro_resolver_ee125f.py` (Folder 10)
- `main_5ae0bd.py` (Folder 10)
- `manifest_0eeebb.py` (Folder 10)
- `media_player_22cc49.py` (Folder 10)
- `media_player_8d1e5c.py` (Folder 10)
- `middleware_52297d.py` (Folder 10)
- `models_bd5f94.py` (Folder 10)
- `networks_c6ab68.py` (Folder 10)
- `notify_1817bd.py` (Folder 10)
- `notify_translations_290c75.py` (Folder 10)
- `package_302a40.py` (Folder 10)
- `predict_dd3e15.py` (Folder 10)
- `pretrained_transformer_indexer_c496a2.py` (Folder 10)
- `pretty_5d5a4b.py` (Folder 10)
- `projected_gradient_descent_d15783.py` (Folder 10)
- `query_render_80e011.py` (Folder 10)
- `queue_3e62fb.py` (Folder 10)
- `random_033806.py` (Folder 10)
- `reporter_7fc9be.py` (Folder 10)
- `ricequant_follower_c10508.py` (Folder 10)
- `schemas_d6d78d.py` (Folder 10)
- `scope_ba10d8.py` (Folder 10)
- `send_to_email_mirror_634f31.py` (Folder 10)
- `sensor_1384fd.py` (Folder 10)
- `sensor_171ae6.py` (Folder 10)
- `sensor_51937f.py` (Folder 10)
- `sensor_64dd1e.py` (Folder 10)
- `sensor_704588.py` (Folder 10)
- `sensor_cb4676.py` (Folder 10)
- `sensor_d50fea.py` (Folder 10)
- `shipment_test_10d843.py` (Folder 10)
- `state_63bb66.py` (Folder 10)
- `steps_1f3861.py` (Folder 10)
- `switch_083dac.py` (Folder 10)
- `switch_1b5725.py` (Folder 10)
- `switch_813f86.py` (Folder 10)
- `test_accessor_fe2995.py` (Folder 10)
- `test_commands_1034dd.py` (Folder 10)
- `test_dataclasses_07986f.py` (Folder 10)
- `test_dependencies_3dbf6b.py` (Folder 10)
- `test_edge_cases_0ecc52.py` (Folder 10)
- `test_frame_plot_plotly_91dfd7.py` (Folder 10)
- `test_freshness_adc6ea.py` (Folder 10)
- `test_homekit_aeba14.py` (Folder 10)
- `test_incremental_schema_fd049d.py` (Folder 10)
- `test_init_5dede4.py` (Folder 10)
- `test_interrupt_6915cc.py` (Folder 10)
- `test_isin_243dba.py` (Folder 10)
- `test_main_0ab538.py` (Folder 10)
- `test_okx_8bd74b.py` (Folder 10)
- `test_pandas_efa602.py` (Folder 10)
- `test_serialize_a3b3a5.py` (Folder 10)
- `test_starrocks_f22333.py` (Folder 10)
- `test_table_5469d2.py` (Folder 10)
- `test_task_e78dda.py` (Folder 10)
- `test_to_latex_569aaf.py` (Folder 10)
- `testing_70aa2b.py` (Folder 10)
- `token_network_5df8c5.py` (Folder 10)
- `type_remotes_f2ab28.py` (Folder 10)
- `typeshed_1ff9b7.py` (Folder 10)
- `update_f13038.py` (Folder 10)
- `user_groups_e1e474.py` (Folder 10)
- `util_f31474.py` (Folder 10)
- `validate_docstrings_2c5bbb.py` (Folder 10)
- `water_heater_44e018.py` (Folder 10)
- `whisk_3be5e7.py` (Folder 10)
- `__init___0ef4ca.py` (Folder 11)
- `__init___69cc40.py` (Folder 11)
- `_checklist_internal_2e9c80.py` (Folder 11)
- `_mass_scanner_653188.py` (Folder 11)
- `_signature_46db1e.py` (Folder 11)
- `accessor_0c5289.py` (Folder 11)
- `adversarial_bias_mitigator_3ad2a0.py` (Folder 11)
- `alarm_control_panel_4aa134.py` (Folder 11)
- `alarm_control_panel_c01c58.py` (Folder 11)
- `alert_f8da8a.py` (Folder 11)
- `api_tests_5c7478.py` (Folder 11)
- `attester_slashings_270ad2.py` (Folder 11)
- `base_30deb6.py` (Folder 11)
- `base_9ced72.py` (Folder 11)
- `base_eadd2e.py` (Folder 11)
- `binary_sensor_2da4f5.py` (Folder 11)
- `bleu_test_22d7f9.py` (Folder 11)
- `boolean_accuracy_test_a96780.py` (Folder 11)
- `channels_with_minimum_balance_ca4241.py` (Folder 11)
- `client_6c9657.py` (Folder 11)
- `climate_c8f819.py` (Folder 11)
- `cloud_run_v2_24c146.py` (Folder 11)
- `color_ef70c6.py` (Folder 11)
- `compression_utils_d50545.py` (Folder 11)
- `connection_f2d984.py` (Folder 11)
- `core_6a4e49.py` (Folder 11)
- `core_6bd602.py` (Folder 11)
- `core_7acc64.py` (Folder 11)
- `data_9b3042.py` (Folder 11)
- `dataset_reader_e7f129.py` (Folder 11)
- `decorator_3d1e96.py` (Folder 11)
- `diff_shades_gha_helper_982e4e.py` (Folder 11)
- `entity_1a0cfb.py` (Folder 11)
- `executor_d0527f.py` (Folder 11)
- `fairscale_fsdp_accelerator_test_c1bd1f.py` (Folder 11)
- `formatters_61cce0.py` (Folder 11)
- `formparsers_534d1c.py` (Folder 11)
- `frozen_dataclass_compat_8ff8c0.py` (Folder 11)
- `function_184992.py` (Folder 11)
- `function_98f82d.py` (Folder 11)
- `geo_location_650a75.py` (Folder 11)
- `group_9c0c4d.py` (Folder 11)
- `grouper_95819c.py` (Folder 11)
- `helpers_4cc809.py` (Folder 11)
- `http_server_4a75c7.py` (Folder 11)
- `humidifier_e2948a.py` (Folder 11)
- `interval_941dbf.py` (Folder 11)
- `light_1bfc0b.py` (Folder 11)
- `light_ae238e.py` (Folder 11)
- `light_f6e8bb.py` (Folder 11)
- `media_player_272d20.py` (Folder 11)
- `media_player_a5ad5b.py` (Folder 11)
- `monitoring_service_7fc586.py` (Folder 11)
- `ops_97a00f.py` (Folder 11)
- `optimize_reports_05f23c.py` (Folder 11)
- `pip-compile-wrapper_4039ba.py` (Folder 11)
- `pystone_d804b9.py` (Folder 11)
- `python_parser_43f40c.py` (Folder 11)
- `PyTorchModelTrainer_91f360.py` (Folder 11)
- `querysets_ea35a5.py` (Folder 11)
- `request_4f1f32.py` (Folder 11)
- `responses_bd7ee5.py` (Folder 11)
- `rest_51609c.py` (Folder 11)
- `runners_bac982.py` (Folder 11)
- `sampled_softmax_loss_ab8f9c.py` (Folder 11)
- `schedules_32080b.py` (Folder 11)
- `securetransport_9efd99.py` (Folder 11)
- `selector_spec_e2bf19.py` (Folder 11)
- `sensor_33e579.py` (Folder 11)
- `sensor_3bb3b2.py` (Folder 11)
- `sensor_687ec5.py` (Folder 11)
- `sensor_6aecf7.py` (Folder 11)
- `sensor_b31c52.py` (Folder 11)
- `sequence_accuracy_test_b352af.py` (Folder 11)
- `server_connectivity_4f3da0.py` (Folder 11)
- `switch_10de32.py` (Folder 11)
- `test_actor_a36a25.py` (Folder 11)
- `test_api_8216ac.py` (Folder 11)
- `test_base_299dbf.py` (Folder 11)
- `test_base_621fba.py` (Folder 11)
- `test_category_67f90a.py` (Folder 11)
- `test_coercion_64efed.py` (Folder 11)
- `test_concurrency_sync_c47702.py` (Folder 11)
- `test_constructors_cc4293.py` (Folder 11)
- `test_encoding_3fd477.py` (Folder 11)
- `test_ex_ante_5d86c9.py` (Folder 11)
- `test_generic_ae8d8a.py` (Folder 11)
- `test_hls_610af1.py` (Folder 11)
- `test_indexing_slow_364644.py` (Folder 11)
- `test_internet_73cc51.py` (Folder 11)
- `test_interval_0c80e1.py` (Folder 11)
- `test_isa_18_2_e8d887.py` (Folder 11)
- `test_join_ff276c.py` (Folder 11)
- `test_merge_index_as_string_675720.py` (Folder 11)
- `test_notify_74a852.py` (Folder 11)
- `test_old_base_c9845a.py` (Folder 11)
- `test_package_4421e1.py` (Folder 11)
- `test_pfs_integration_b86f10.py` (Folder 11)
- `test_round_0a4ab4.py` (Folder 11)
- `test_runner_66f090.py` (Folder 11)
- `test_strategy_helpers_2175ad.py` (Folder 11)
- `test_support_views_55d8a9.py` (Folder 11)
- `test_timestamp_a9efbe.py` (Folder 11)
- `test_types_d2ef74.py` (Folder 11)
- `test_validate_call_bfd6c7.py` (Folder 11)
- `test_web_websocket_389e12.py` (Folder 11)
- `todo_c5f9bc.py` (Folder 11)
- `token_8cf4ee.py` (Folder 11)
- `token_characters_indexer_4a2ab1.py` (Folder 11)
- `token_network_registry_0d90a4.py` (Folder 11)
- `toolkit_test_11b0e6.py` (Folder 11)
- `tracing_98653c.py` (Folder 11)
- `transformer_text_field_2fd797.py` (Folder 11)
- `user_topics_1dd5f1.py` (Folder 11)
- `utils_21e64d.py` (Folder 11)
- `validation_a5e875.py` (Folder 11)
- `waiting_fc6a47.py` (Folder 11)
- `web_exceptions_bbf405.py` (Folder 11)
- `web_urldispatcher_77a3cf.py` (Folder 11)
- `websocket_api_aac75a.py` (Folder 11)
- `__init___272668.py` (Folder 12)
- `__init___2d57a9.py` (Folder 12)
- `__init___705fad.py` (Folder 12)
- `__init___8ac541.py` (Folder 12)
- `__init___916118.py` (Folder 12)
- `__init___96a5ba.py` (Folder 12)
- `__init___b92714.py` (Folder 12)
- `_mock_val_ser_7914af.py` (Folder 12)
- `_validate_call_b8a4cf.py` (Folder 12)
- `acked_by_ce084a.py` (Folder 12)
- `assist_satellite_7b41c3.py` (Folder 12)
- `attachment_scores_test_b605c1.py` (Folder 12)
- `base_292eb6.py` (Folder 12)
- `base_a0b3af.py` (Folder 12)
- `base_b5c467.py` (Folder 12)
- `base_labeler_c2b04f.py` (Folder 12)
- `binary_sensor_46da6d.py` (Folder 12)
- `binary_sensor_71f3af.py` (Folder 12)
- `boundary_attack_c6f415.py` (Folder 12)
- `camera_5fe39c.py` (Folder 12)
- `cast_b125eb.py` (Folder 12)
- `client_proto_8b80d0.py` (Folder 12)
- `common_d2ff1b.py` (Folder 12)
- `common_f50d87.py` (Folder 12)
- `completion_8d3a60.py` (Folder 12)
- `conftest_f1c9f3.py` (Folder 12)
- `connection_92ca1e.py` (Folder 12)
- `coordinator_fa65bb.py` (Folder 12)
- `cover_fcf81d.py` (Folder 12)
- `curves_481429.py` (Folder 12)
- `dataset_store_87323b.py` (Folder 12)
- `deepfool_7d0a6a.py` (Folder 12)
- `discovery_c67402.py` (Folder 12)
- `entity_547f2b.py` (Folder 12)
- `entity_9249a5.py` (Folder 12)
- `entity_b4f398.py` (Folder 12)
- `entity_fb6002.py` (Folder 12)
- `escalate_8b7ec7.py` (Folder 12)
- `event_logger_tests_21ea29.py` (Folder 12)
- `events_e1f80f.py` (Folder 12)
- `freqai_rl_test_strat_37a1df.py` (Folder 12)
- `gen_runner_a44ef9.py` (Folder 12)
- `helper_de3781.py` (Folder 12)
- `hop_skip_jump_d70890.py` (Folder 12)
- `html_5c3937.py` (Folder 12)
- `image_processing_3ab1ef.py` (Folder 12)
- `instaloadercontext_7cfe14.py` (Folder 12)
- `interleaving_dataset_reader_c0729d.py` (Folder 12)
- `jediusages_96bc8b.py` (Folder 12)
- `json_7c2407.py` (Folder 12)
- `loss_465239.py` (Folder 12)
- `media_player_08a831.py` (Folder 12)
- `media_player_391475.py` (Folder 12)
- `media_player_9a7ce8.py` (Folder 12)
- `migrations_5154b3.py` (Folder 12)
- `model_d392e2.py` (Folder 12)
- `multi_586637.py` (Folder 12)
- `multipart_07595c.py` (Folder 12)
- `numbers_5dcc4a.py` (Folder 12)
- `oauth2_610d7e.py` (Folder 12)
- `optimiser_abeb97.py` (Folder 12)
- `parameters_38dd02.py` (Folder 12)
- `partial_619858.py` (Folder 12)
- `pep_570_5b7cab.py` (Folder 12)
- `pipeline_4ce07e.py` (Folder 12)
- `presence_ce7e57.py` (Folder 12)
- `pretrained_transformer_tokenizer_f82621.py` (Folder 12)
- `pysteroids_ffbac0.py` (Folder 12)
- `query_context_factory_3e6211.py` (Folder 12)
- `query_object_factory_d095d1.py` (Folder 12)
- `registry_e23820.py` (Folder 12)
- `response_0a5943.py` (Folder 12)
- `rocksdb_2826c4.py` (Folder 12)
- `sas_xport_0b1398.py` (Folder 12)
- `sensor_2394c2.py` (Folder 12)
- `sensor_32c6be.py` (Folder 12)
- `sensor_4b2742.py` (Folder 12)
- `sensor_8c7ef1.py` (Folder 12)
- `slack_tests_5e5a8c.py` (Folder 12)
- `sql_648076.py` (Folder 12)
- `sql_lab_a1d120.py` (Folder 12)
- `stack_621148.py` (Folder 12)
- `style_render_e06ec7.py` (Folder 12)
- `system_55d175.py` (Folder 12)
- `test_base_f26fe7.py` (Folder 12)
- `test_boxplot_method_abfc5c.py` (Folder 12)
- `test_build_signature_ea8663.py` (Folder 12)
- `test_categorical_e04b8e.py` (Folder 12)
- `test_clique_consensus_e99cfa.py` (Folder 12)
- `test_constructors_22c366.py` (Folder 12)
- `test_convert_dtypes_c20223.py` (Folder 12)
- `test_date_a04db1.py` (Folder 12)
- `test_graph_selection_f68296.py` (Folder 12)
- `test_indices_7db737.py` (Folder 12)
- `test_init_e32ce0.py` (Folder 12)
- `test_initiatorstate_f29784.py` (Folder 12)
- `test_invalid_arg_f69c9f.py` (Folder 12)
- `test_json_schema_0712a3.py` (Folder 12)
- `test_notifications_endpoint_08d1aa.py` (Folder 12)
- `test_pairlist_c95c81.py` (Folder 12)
- `test_period_index_9381a7.py` (Folder 12)
- `test_precise_shrinking_51b995.py` (Folder 12)
- `test_session_d27610.py` (Folder 12)
- `test_stat_reductions_69f5b2.py` (Folder 12)
- `test_to_csv_c73ed7.py` (Folder 12)
- `timedeltas_bd7047.py` (Folder 12)
- `transactions_739961.py` (Folder 12)
- `transfer_83a674.py` (Folder 12)
- `types_ddfd4c.py` (Folder 12)
- `upgrade_a9a989.py` (Folder 12)
- `util_c87ff9.py` (Folder 12)
- `utils_51c496.py` (Folder 12)
- `utils_71116c.py` (Folder 12)
- `weather_26d8e5.py` (Folder 12)
- `web_response_49adee.py` (Folder 12)
- `xpbase_ec8d7a.py` (Folder 12)
- `__init___417be1.py` (Folder 13)
- `__init___54e0bc.py` (Folder 13)
- `__init___93171f.py` (Folder 13)
- `__init___ade613.py` (Folder 13)
- `__init___c1e2a2.py` (Folder 13)
- `__init___d275cc.py` (Folder 13)
- `_mixins_640936.py` (Folder 13)
- `_typing_3c4262.py` (Folder 13)
- `_version_d0367b.py` (Folder 13)
- `aiohttp_0b8820.py` (Folder 13)
- `api_1ca83f.py` (Folder 13)
- `base_a2ef3c.py` (Folder 13)
- `BasePyTorchClassifier_55eaf8.py` (Folder 13)
- `binance_502cb1.py` (Folder 13)
- `binary_sensor_4e2b33.py` (Folder 13)
- `binary_sensor_b774df.py` (Folder 13)
- `blackout_da4461.py` (Folder 13)
- `block_schemas_d88c89.py` (Folder 13)
- `boxplot_f690b9.py` (Folder 13)
- `button_11a8b2.py` (Folder 13)
- `button_2ec72a.py` (Folder 13)
- `calendar_819d29.py` (Folder 13)
- `client_819f65.py` (Folder 13)
- `client_e5a475.py` (Folder 13)
- `common_1f9133.py` (Folder 13)
- `common_4aeea7.py` (Folder 13)
- `config_c41598.py` (Folder 13)
- `config_edac54.py` (Folder 13)
- `conftest_89a5b3.py` (Folder 13)
- `conll2003_71bf0e.py` (Folder 13)
- `controllers_c0fa43.py` (Folder 13)
- `cover_cd851c.py` (Folder 13)
- `diff_011ffc.py` (Folder 13)
- `edge_positioning_82e827.py` (Folder 13)
- `entity_163783.py` (Folder 13)
- `entity_45357b.py` (Folder 13)
- `entity_86b8a0.py` (Folder 13)
- `errors_5e09ac.py` (Folder 13)
- `execute_6ea9d9.py` (Folder 13)
- `expressions_5f4f2d.py` (Folder 13)
- `fixtures_82d048.py` (Folder 13)
- `fixtures_f8948e.py` (Folder 13)
- `flow_engine_887be2.py` (Folder 13)
- `freqai_interface_c93cd2.py` (Folder 13)
- `FreqaiExampleStrategy_4d11fd.py` (Folder 13)
- `gen_attack_5c1da7.py` (Folder 13)
- `heartbeat_5b669e.py` (Folder 13)
- `helper_91cf7a.py` (Folder 13)
- `hooks_bd685b.py` (Folder 13)
- `http_adapters_298d8d.py` (Folder 13)
- `image_loader_a0ea65.py` (Folder 13)
- `interval_216808.py` (Folder 13)
- `jedi_handler_dbc20a.py` (Folder 13)
- `lambda_dataset_8c1015.py` (Folder 13)
- `light_250268.py` (Folder 13)
- `linegen_ea8858.py` (Folder 13)
- `linter_43a04e.py` (Folder 13)
- `messages_469ee9.py` (Folder 13)
- `models_8da4c8.py` (Folder 13)
- `multilabel_field_d11efd.py` (Folder 13)
- `number_af29a9.py` (Folder 13)
- `parser_deb8f1.py` (Folder 13)
- `patchers_000251.py` (Folder 13)
- `pystone_orig_03dfcb.py` (Folder 13)
- `query_context_processor_356fe7.py` (Folder 13)
- `red_test_90d8dd.py` (Folder 13)
- `redis_925650.py` (Folder 13)
- `registry_6e5f77.py` (Folder 13)
- `results_6ab672.py` (Folder 13)
- `schema_config_entry_flow_895524.py` (Folder 13)
- `sensor_000a20.py` (Folder 13)
- `sensor_41340f.py` (Folder 13)
- `sensor_4165dd.py` (Folder 13)
- `sensor_51b8a7.py` (Folder 13)
- `sensor_c2feed.py` (Folder 13)
- `sensor_c43e72.py` (Folder 13)
- `sensor_device_4e1fbf.py` (Folder 13)
- `sensor_e182c2.py` (Folder 13)
- `sensor_f28e92.py` (Folder 13)
- `sensor_fdc120.py` (Folder 13)
- `simulationresult_ec673f.py` (Folder 13)
- `six_75db7b.py` (Folder 13)
- `speaker_65ea42.py` (Folder 13)
- `starrocks_a24cf8.py` (Folder 13)
- `statistics_meta_5a6ed1.py` (Folder 13)
- `swagger_c55f37.py` (Folder 13)
- `switch_319ae9.py` (Folder 13)
- `switch_71a6f5.py` (Folder 13)
- `switch_c85c80.py` (Folder 13)
- `task_runners_1245a4.py` (Folder 13)
- `templating_b18af2.py` (Folder 13)
- `test_collections_7ffee6.py` (Folder 13)
- `test_commands_20db9a.py` (Folder 13)
- `test_compression_7a6a44.py` (Folder 13)
- `test_construction_01d413.py` (Folder 13)
- `test_constructors_e8ae8e.py` (Folder 13)
- `test_deposit_93dc29.py` (Folder 13)
- `test_ecs_worker_42cad9.py` (Folder 13)
- `test_exchange_utils_066f92.py` (Folder 13)
- `test_flows_9f83e6.py` (Folder 13)
- `test_formats_c7732b.py` (Folder 13)
- `test_incremental_on_schema_change_ffcc97.py` (Folder 13)
- `test_init_873ad0.py` (Folder 13)
- `test_init_8d5e26.py` (Folder 13)
- `test_interval_tree_05bf63.py` (Folder 13)
- `test_parse_89178c.py` (Folder 13)
- `test_purge_v32_schema_a4bc69.py` (Folder 13)
- `test_round_faa621.py` (Folder 13)
- `test_ujson_e78381.py` (Folder 13)
- `timerange_d18ed2.py` (Folder 13)
- `tracing_fe42d6.py` (Folder 13)
- `transformer_embeddings_ca9dbe.py` (Folder 13)
- `ulauncher_app_b6c22b.py` (Folder 13)
- `unigram_recall_test_83abd5.py` (Folder 13)
- `unit_tests_627911.py` (Folder 13)
- `utils_03a814.py` (Folder 13)
- `utils_3f6c97.py` (Folder 13)
- `utils_f7b84e.py` (Folder 13)
- `vagrant_71cd9c.py` (Folder 13)
- `validators_1837b8.py` (Folder 13)
- `websocket_9eba38.py` (Folder 13)
- `websocket_api_f30f86.py` (Folder 13)
- `wsgi_30f67e.py` (Folder 13)
- `__init___49fc81.py` (Folder 14)
- `__init___4c124d.py` (Folder 14)
- `__init___70582f.py` (Folder 14)
- `__init___c1477c.py` (Folder 14)
- `__init___dc7584.py` (Folder 14)
- `_datalayers_fb5039.py` (Folder 14)
- `actions_622ea4.py` (Folder 14)
- `anaconda_pep8_08d915.py` (Folder 14)
- `async_query_manager_ff88dd.py` (Folder 14)
- `base_e2d6f6.py` (Folder 14)
- `binary_sensor_7dd664.py` (Folder 14)
- `binary_sensor_c45748.py` (Folder 14)
- `camera_cb6050.py` (Folder 14)
- `camera_ed1077.py` (Folder 14)
- `check-env_44d55f.py` (Folder 14)
- `checker_7dee21.py` (Folder 14)
- `cli_605725.py` (Folder 14)
- `client_575ed4.py` (Folder 14)
- `clients_564827.py` (Folder 14)
- `climate_3d3d8e.py` (Folder 14)
- `cloud_run_463bed.py` (Folder 14)
- `commands_98e404.py` (Folder 14)
- `common_bbc0b0.py` (Folder 14)
- `common_e59185.py` (Folder 14)
- `config_ae366e.py` (Folder 14)
- `conftest_3798fb.py` (Folder 14)
- `connection_6f580e.py` (Folder 14)
- `consumer_fa3d87.py` (Folder 14)
- `controls_5fc8a5.py` (Folder 14)
- `coordinator_bb51e9.py` (Folder 14)
- `cover_a9a7cf.py` (Folder 14)
- `custody_a914e5.py` (Folder 14)
- `data_0433fa.py` (Folder 14)
- `datatree_65bfb0.py` (Folder 14)
- `db_schema_22_6df366.py` (Folder 14)
- `ddp_accelerator_0b7bea.py` (Folder 14)
- `decorators_464c62.py` (Folder 14)
- `device_tracker_568db5.py` (Folder 14)
- `encoding_c1153a.py` (Folder 14)
- `entity_47b6f9.py` (Folder 14)
- `entity_99a193.py` (Folder 14)
- `entity_e38773.py` (Folder 14)
- `entity_loader_12cd6d.py` (Folder 14)
- `execution_855aa0.py` (Folder 14)
- `fields_fc8b25.py` (Folder 14)
- `humidifier_8c532f.py` (Folder 14)
- `jsonserver_34dd2d.py` (Folder 14)
- `legacy_148a53.py` (Folder 14)
- `light_7346d9.py` (Folder 14)
- `log_api_tests_2cbd8d.py` (Folder 14)
- `markdown_extension_f5018d.py` (Folder 14)
- `media_player_5b2733.py` (Folder 14)
- `metadata_6d0eb8.py` (Folder 14)
- `models_7ff78a.py` (Folder 14)
- `multitask_24dbfa.py` (Folder 14)
- `navigator_watcher_5f9fc6.py` (Folder 14)
- `normalizer_c4fa95.py` (Folder 14)
- `note_76679d.py` (Folder 14)
- `pairlock_middleware_5f3e93.py` (Folder 14)
- `poolmanager_263b1b.py` (Folder 14)
- `poolmanager_9d2f2e.py` (Folder 14)
- `pretrained_transformer_mismatched_indexer_f25f15.py` (Folder 14)
- `pytorch_transformer_wrapper_6a1b05.py` (Folder 14)
- `pytree_adefa1.py` (Folder 14)
- `queue_9bbba9.py` (Folder 14)
- `rekognition_c54474.py` (Folder 14)
- `reshape_cabe3c.py` (Folder 14)
- `sensor_1c6e89.py` (Folder 14)
- `sensor_393e9c.py` (Folder 14)
- `sensor_4258dc.py` (Folder 14)
- `sensor_5837db.py` (Folder 14)
- `sensor_6b1a2d.py` (Folder 14)
- `sensor_7d888f.py` (Folder 14)
- `sensor_db85c4.py` (Folder 14)
- `sensor_db8e1f.py` (Folder 14)
- `sensor_e720bf.py` (Folder 14)
- `sensor_e7d661.py` (Folder 14)
- `sensor_f11fa0.py` (Folder 14)
- `server_b8ff37.py` (Folder 14)
- `settings_004992.py` (Folder 14)
- `signals_ff12b9.py` (Folder 14)
- `signature_02ba12.py` (Folder 14)
- `smartcontracts_ab65b2.py` (Folder 14)
- `spacy_tokenizer_edae48.py` (Folder 14)
- `sphere_97cb66.py` (Folder 14)
- `sql_parse_fe229f.py` (Folder 14)
- `sqlite_8d53fd.py` (Folder 14)
- `strings_ba32d0.py` (Folder 14)
- `switch_03d631.py` (Folder 14)
- `switch_11f4bc.py` (Folder 14)
- `switch_a299a8.py` (Folder 14)
- `switch_d3c2da.py` (Folder 14)
- `test_bar_9978a0.py` (Folder 14)
- `test_configuration_798036.py` (Folder 14)
- `test_construction_223323.py` (Folder 14)
- `test_digest_a03eb9.py` (Folder 14)
- `test_dtypes_e2af21.py` (Folder 14)
- `test_hist_box_by_07b6c3.py` (Folder 14)
- `test_hooks_5c2427.py` (Folder 14)
- `test_img_util_826a0b.py` (Folder 14)
- `test_jupyter_69b856.py` (Folder 14)
- `test_media_source_357294.py` (Folder 14)
- `test_microbatch_04817a.py` (Folder 14)
- `test_odswriter_dd1643.py` (Folder 14)
- `test_optimizerlib_7529cc.py` (Folder 14)
- `test_partial_parsing_719695.py` (Folder 14)
- `test_presto_ffe855.py` (Folder 14)
- `test_semantic_manifest_0ca206.py` (Folder 14)
- `test_steps_cf24e2.py` (Folder 14)
- `test_tables_ba47a0.py` (Folder 14)
- `testing_1ecbec.py` (Folder 14)
- `thumbnail_ae48ab.py` (Folder 14)
- `tools_632c06.py` (Folder 14)
- `trino_6b3cce.py` (Folder 14)
- `tts_834735.py` (Folder 14)
- `update_96e467.py` (Folder 14)
- `url_77bbe4.py` (Folder 14)
- `util_04035d.py` (Folder 14)
- `util_c4e661.py` (Folder 14)
- `web_server_a074c2.py` (Folder 14)
- `websocket_api_e94b97.py` (Folder 14)
- `webtrader_b07629.py` (Folder 14)
- `0531_convert_most_ids_to_bigints_841260.py` (Folder 15)
- `__init___4809f5.py` (Folder 15)
- `__init___a5232e.py` (Folder 15)
- `__init___ae6858.py` (Folder 15)
- `__init___c981d8.py` (Folder 15)
- `_array_helpers_54724c.py` (Folder 15)
- `_generate_schema_391a7e.py` (Folder 15)
- `_prompts_6a5c3e.py` (Folder 15)
- `addressee_265b46.py` (Folder 15)
- `aenum_f94dc8.py` (Folder 15)
- `air_quality_ab8c69.py` (Folder 15)
- `alarm_control_panel_546dbe.py` (Folder 15)
- `app_9acae6.py` (Folder 15)
- `appengine_d9dc66.py` (Folder 15)
- `array_1d8ad4.py` (Folder 15)
- `arraylike_473d9a.py` (Folder 15)
- `auth_0ecea0.py` (Folder 15)
- `binary_sensor_39cce2.py` (Folder 15)
- `bybit_f190ec.py` (Folder 15)
- `calendar_211164.py` (Folder 15)
- `checker_a02381.py` (Folder 15)
- `climate_0e5162.py` (Folder 15)
- `climate_a5fa26.py` (Folder 15)
- `collection_a8ceb1.py` (Folder 15)
- `collections_139bce.py` (Folder 15)
- `common_4a17c0.py` (Folder 15)
- `config_flow_999455.py` (Folder 15)
- `conftest_51a703.py` (Folder 15)
- `conftest_b8b972.py` (Folder 15)
- `conftest_d110aa.py` (Folder 15)
- `construction_0cbb84.py` (Folder 15)
- `core_214ccb.py` (Folder 15)
- `core_753085.py` (Folder 15)
- `db_0b7e99.py` (Folder 15)
- `deprecated_86e1b2.py` (Folder 15)
- `diff_65255d.py` (Folder 15)
- `driver_94aaee.py` (Folder 15)
- `ecs_3fb000.py` (Folder 15)
- `fan_7d4617.py` (Folder 15)
- `freqai_test_classifier_bd29bc.py` (Folder 15)
- `FreqaiExampleHybridStrategy_04bd2f.py` (Folder 15)
- `freqtradebot_43234d.py` (Folder 15)
- `geo_location_c37acf.py` (Folder 15)
- `helpers_062f96.py` (Folder 15)
- `helpers_b04c1d.py` (Folder 15)
- `hotflip_66083c.py` (Folder 15)
- `html_branches_9ad2ad.py` (Folder 15)
- `image_processing_d4a5a8.py` (Folder 15)
- `isa_18_2_b9e836.py` (Folder 15)
- `jinja_17d794.py` (Folder 15)
- `key_value_68efd3.py` (Folder 15)
- `lazy_value_18eb7d.py` (Folder 15)
- `makemessages_b575f4.py` (Folder 15)
- `media_player_6853b0.py` (Folder 15)
- `message_edit_6be856.py` (Folder 15)
- `model_cb731f.py` (Folder 15)
- `number_6fca76.py` (Folder 15)
- `number_c8a6fb.py` (Folder 15)
- `numeric_40361d.py` (Folder 15)
- `param_cbed36.py` (Folder 15)
- `payload_c043e8.py` (Folder 15)
- `pong_48dbec.py` (Folder 15)
- `py2md_98a5cb.py` (Folder 15)
- `pyopenssl_014c19.py` (Folder 15)
- `remote_billing_page_79cc06.py` (Folder 15)
- `resources_83f462.py` (Folder 15)
- `router_0f9079.py` (Folder 15)
- `routing_cbdb3d.py` (Folder 15)
- `scoring_bded35.py` (Folder 15)
- `sensor_07dce7.py` (Folder 15)
- `sensor_0a8e9e.py` (Folder 15)
- `sensor_1c2083.py` (Folder 15)
- `sensor_2e8c82.py` (Folder 15)
- `sensor_3731ea.py` (Folder 15)
- `sensor_3f0300.py` (Folder 15)
- `sensor_935eef.py` (Folder 15)
- `sensor_a8d27b.py` (Folder 15)
- `sensor_bb5bae.py` (Folder 15)
- `sensor_d95271.py` (Folder 15)
- `sentry_ed3e24.py` (Folder 15)
- `states_ff542b.py` (Folder 15)
- `steps_6793e6.py` (Folder 15)
- `stoppers_8ab3a2.py` (Folder 15)
- `tcpclient_c8e77d.py` (Folder 15)
- `test_app_95245f.py` (Folder 15)
- `test_clip_1e0363.py` (Folder 15)
- `test_config_entry_43b322.py` (Folder 15)
- `test_contract_call_6ddf95.py` (Folder 15)
- `test_doris_230b68.py` (Folder 15)
- `test_drop_duplicates_ed341e.py` (Folder 15)
- `test_frame_subplots_be5b29.py` (Folder 15)
- `test_from_dtype_806539.py` (Folder 15)
- `test_gcs_fb4ad5.py` (Folder 15)
- `test_include_router_defaults_overrides_afc221.py` (Folder 15)
- `test_internals_d33223.py` (Folder 15)
- `test_liboffsets_a4381d.py` (Folder 15)
- `test_london_5cebb4.py` (Folder 15)
- `test_moments_consistency_rolling_0c7eb0.py` (Folder 15)
- `test_mysql_b75555.py` (Folder 15)
- `test_on_attestation_3db217.py` (Folder 15)
- `test_parser_da9792.py` (Folder 15)
- `test_setops_5584ad.py` (Folder 15)
- `test_starters_667c4a.py` (Folder 15)
- `test_subclass_cc44d0.py` (Folder 15)
- `test_timedelta_range_f3b73d.py` (Folder 15)
- `test_to_datetime_ac3e83.py` (Folder 15)
- `test_unit_conversion_d2dfd2.py` (Folder 15)
- `thread_78e150.py` (Folder 15)
- `tokenize_204134.py` (Folder 15)
- `tooltips_4371f8.py` (Folder 15)
- `utils_5a157a.py` (Folder 15)
- `utils_8f085b.py` (Folder 15)
- `vad_69bc6f.py` (Folder 15)
- `validate_a55939.py` (Folder 15)
- `versioneer_4c3459.py` (Folder 15)
- `viz_177254.py` (Folder 15)
- `water_heater_2676b6.py` (Folder 15)
- `water_heater_ccfba1.py` (Folder 15)
- `__init___07d9ec.py` (Folder 16)
- `__init___404213.py` (Folder 16)
- `__init___53e8bc.py` (Folder 16)
- `_doctools_2a0387.py` (Folder 16)
- `_generics_9112b1.py` (Folder 16)
- `_odswriter_123864.py` (Folder 16)
- `_openpyxl_60b4ed.py` (Folder 16)
- `activity_5ae188.py` (Folder 16)
- `aiohttp_client_6b2f4d.py` (Folder 16)
- `aiohttp_d6c061.py` (Folder 16)
- `air_quality_44856d.py` (Folder 16)
- `api_return_values_table_generator_45d9fa.py` (Folder 16)
- `area_registry_5d0c96.py` (Folder 16)
- `array_api_954234.py` (Folder 16)
- `assist_satellite_9dcb75.py` (Folder 16)
- `base_73fb43.py` (Folder 16)
- `basemodel_eq_performance_5b0d75.py` (Folder 16)
- `binary_sensor_38593e.py` (Folder 16)
- `binary_sensor_d6298b.py` (Folder 16)
- `callbacks_221fdc.py` (Folder 16)
- `carlini_wagner_641b5f.py` (Folder 16)
- `catalogs_ae5846.py` (Folder 16)
- `cli_aa35f8.py` (Folder 16)
- `climate_c46a91.py` (Folder 16)
- `climate_c4a00f.py` (Folder 16)
- `comments6_b4c7b6.py` (Folder 16)
- `common_source_setup_ccf6d3.py` (Folder 16)
- `compiler1_c7f07f.py` (Folder 16)
- `configuration_ab5265.py` (Folder 16)
- `conftest_06275c.py` (Folder 16)
- `conftest_224ed3.py` (Folder 16)
- `conftest_ee8a90.py` (Folder 16)
- `control_e5bc2a.py` (Folder 16)
- `coordinator_0bc28f.py` (Folder 16)
- `coordinator_7baa19.py` (Folder 16)
- `coordinator_f89a5c.py` (Folder 16)
- `core_1410fd.py` (Folder 16)
- `core_f49d0c.py` (Folder 16)
- `cover_376334.py` (Folder 16)
- `database_5bbb39.py` (Folder 16)
- `db_schema_18_7a6f03.py` (Folder 16)
- `debug_info_f902e5.py` (Folder 16)
- `dtypes_dd7589.py` (Folder 16)
- `dynamic_params_5296db.py` (Folder 16)
- `ead_50dc22.py` (Folder 16)
- `encoder_base_ec641d.py` (Folder 16)
- `entity_a41728.py` (Folder 16)
- `entity_platform_bbf71d.py` (Folder 16)
- `factories_93a9dd.py` (Folder 16)
- `fairscale_fsdp_accelerator_e2fda0.py` (Folder 16)
- `featherdatahandler_48f824.py` (Folder 16)
- `freqai_test_multimodel_classifier_strat_74a532.py` (Folder 16)
- `gsheets_10a4ee.py` (Folder 16)
- `helpers_4932dc.py` (Folder 16)
- `home_349a7c.py` (Folder 16)
- `host_37696c.py` (Folder 16)
- `hyperopt_tools_3fad83.py` (Folder 16)
- `image_91d679.py` (Folder 16)
- `jsl_039295.py` (Folder 16)
- `layers_44aa42.py` (Folder 16)
- `light_4faab7.py` (Folder 16)
- `light_cd075f.py` (Folder 16)
- `light_d31785.py` (Folder 16)
- `log_config_fb42e8.py` (Folder 16)
- `media_player_b0f7c2.py` (Folder 16)
- `mediation_fees_8611a5.py` (Folder 16)
- `migrate_c6f77f.py` (Folder 16)
- `migration_274a81.py` (Folder 16)
- `minio_helper_ffe998.py` (Folder 16)
- `number_e901fd.py` (Folder 16)
- `question_answering_suite_c14a3b.py` (Folder 16)
- `read_files_0de626.py` (Folder 16)
- `recorder_650626.py` (Folder 16)
- `reject_06c37d.py` (Folder 16)
- `runner_1ad9f2.py` (Folder 16)
- `select_a77de5.py` (Folder 16)
- `sensor_14ed48.py` (Folder 16)
- `sensor_15ddcf.py` (Folder 16)
- `sensor_2062f9.py` (Folder 16)
- `sensor_341440.py` (Folder 16)
- `sensor_5d625f.py` (Folder 16)
- `sensor_875fbc.py` (Folder 16)
- `sensor_bd2e29.py` (Folder 16)
- `sensor_cc9362.py` (Folder 16)
- `sensor_efafae.py` (Folder 16)
- `six_36a467.py` (Folder 16)
- `sql_json_executer_85737e.py` (Folder 16)
- `stress_test_transfers_aeecfe.py` (Folder 16)
- `switch_0cde31.py` (Folder 16)
- `sync_backend_1b9465.py` (Folder 16)
- `syntax_tree_78a1cf.py` (Folder 16)
- `take_7704c4.py` (Folder 16)
- `task_runners_62fb9b.py` (Folder 16)
- `test_catalog_230917.py` (Folder 16)
- `test_cli_hooks_3cceea.py` (Folder 16)
- `test_denodo_4cd030.py` (Folder 16)
- `test_env_var_deprecations_894c4e.py` (Folder 16)
- `test_imports_3305a5.py` (Folder 16)
- `test_index_503dc5.py` (Folder 16)
- `test_inquisitor_e1b111.py` (Folder 16)
- `test_json_table_schema_ext_dtype_03d9d8.py` (Folder 16)
- `test_media_player_579798.py` (Folder 16)
- `test_openpyxl_05fcc3.py` (Folder 16)
- `test_overlaps_cda380.py` (Folder 16)
- `test_proxies_b2ba05.py` (Folder 16)
- `test_rank_d0371a.py` (Folder 16)
- `test_registry_c9200e.py` (Folder 16)
- `test_seq_copy_int_d9450d.py` (Folder 16)
- `test_statsd_afaa78.py` (Folder 16)
- `test_suggest_43eeb2.py` (Folder 16)
- `test_tokenize_410db8.py` (Folder 16)
- `test_type_adapter_97b931.py` (Folder 16)
- `tokenize_24b106.py` (Folder 16)
- `translation_dba571.py` (Folder 16)
- `typing_007d2b.py` (Folder 16)
- `utils_b0aeeb.py` (Folder 16)
- `views_61a9fc.py` (Folder 16)
- `violations_cd5822.py` (Folder 16)
- `wal_496e40.py` (Folder 16)
- `__init___886e1c.py` (Folder 17)
- `__init___9e75d4.py` (Folder 17)
- `__init___b55467.py` (Folder 17)
- `__init___e58d5f.py` (Folder 17)
- `alarm_control_panel_397cee.py` (Folder 17)
- `align_a1b6e3.py` (Folder 17)
- `anaconda_mypy_7e3d77.py` (Folder 17)
- `base_4e93c1.py` (Folder 17)
- `bias_direction_wrappers_2ab1fb.py` (Folder 17)
- `bigquery_22dc4f.py` (Folder 17)
- `binary_sensor_c09c5d.py` (Folder 17)
- `binary_sensor_d63eaf.py` (Folder 17)
- `blueprints_a0a4a9.py` (Folder 17)
- `cache_7cc469.py` (Folder 17)
- `cache_961e29.py` (Folder 17)
- `calendar_e8e7db.py` (Folder 17)
- `callback_14d3b1.py` (Folder 17)
- `camera_24f483.py` (Folder 17)
- `changelog_22a926.py` (Folder 17)
- `checks_cda4fd.py` (Folder 17)
- `client_79ecdd.py` (Folder 17)
- `climate_badc68.py` (Folder 17)
- `cloud_run_v2_26c714.py` (Folder 17)
- `commands_764f18.py` (Folder 17)
- `conftest_fea4e8.py` (Folder 17)
- `context_48db84.py` (Folder 17)
- `coordinator_6451a9.py` (Folder 17)
- `coordinator_a798bf.py` (Folder 17)
- `cover_14aabc.py` (Folder 17)
- `elmo_indexer_111385.py` (Folder 17)
- `encoding_8cbe7e.py` (Folder 17)
- `engine_a3dedd.py` (Folder 17)
- `entity_369eaa.py` (Folder 17)
- `escape_d8d166.py` (Folder 17)
- `exceptions_34f7ac.py` (Folder 17)
- `exceptions_8d3eeb.py` (Folder 17)
- `execute_test_6a1c8e.py` (Folder 17)
- `extension_runtime_6679fc.py` (Folder 17)
- `filters_a79ada.py` (Folder 17)
- `freqai_test_multimodel_strat_b37a4b.py` (Folder 17)
- `game_1baf0c.py` (Folder 17)
- `generate_cli_docs_b09af9.py` (Folder 17)
- `generate_email_a0cfd9.py` (Folder 17)
- `geo_location_e1d2b4.py` (Folder 17)
- `grammar_96eff6.py` (Folder 17)
- `guess_8066e7.py` (Folder 17)
- `helpers_e73aba.py` (Folder 17)
- `holiday_1974b0.py` (Folder 17)
- `indexing_b000ec.py` (Folder 17)
- `indexing_engines_bfb33a.py` (Folder 17)
- `jsondatahandler_96667e.py` (Folder 17)
- `light_3a2cf8.py` (Folder 17)
- `light_c96c0e.py` (Folder 17)
- `logger_f99b27.py` (Folder 17)
- `makedoc_4357c2.py` (Folder 17)
- `manager_c291de.py` (Folder 17)
- `module_b33ece.py` (Folder 17)
- `mutations_def7ac.py` (Folder 17)
- `nanops_961886.py` (Folder 17)
- `node_2d78d3.py` (Folder 17)
- `notify_c0a050.py` (Folder 17)
- `online_f12d20.py` (Folder 17)
- `optimistic_sync_7e83e4.py` (Folder 17)
- `page_752b64.py` (Folder 17)
- `parser_utils_3b717f.py` (Folder 17)
- `pipeline_2e0d3e.py` (Folder 17)
- `portico_6e827d.py` (Folder 17)
- `purge_77fd87.py` (Folder 17)
- `pytorch_seq2vec_wrapper_ae605d.py` (Folder 17)
- `realm_linkifiers_90aa4a.py` (Folder 17)
- `router_b20077.py` (Folder 17)
- `sampler_d1337e.py` (Folder 17)
- `scheduled_messages_4b56ca.py` (Folder 17)
- `selectn_fe317c.py` (Folder 17)
- `send_custom_email_c1af50.py` (Folder 17)
- `sensor_03a6d2.py` (Folder 17)
- `sensor_28c444.py` (Folder 17)
- `sensor_653267.py` (Folder 17)
- `sensor_818a5d.py` (Folder 17)
- `sensor_87caf9.py` (Folder 17)
- `sensor_b832ec.py` (Folder 17)
- `sensor_e47032.py` (Folder 17)
- `single_id_token_indexer_c980de.py` (Folder 17)
- `siren_d075cc.py` (Folder 17)
- `spatial_attack_d8e9cb.py` (Folder 17)
- `sublime_b6199e.py` (Folder 17)
- `task_runners_ba6787.py` (Folder 17)
- `telegrams_6b80b7.py` (Folder 17)
- `test_commands_541150.py` (Folder 17)
- `test_config_flow_f9b0c4.py` (Folder 17)
- `test_deposit_transition_1d40b5.py` (Folder 17)
- `test_logging_8b4caf.py` (Folder 17)
- `test_micropkg_pull_c2e238.py` (Folder 17)
- `test_parquet_afc973.py` (Folder 17)
- `test_process_participation_flag_updates_24798c.py` (Folder 17)
- `test_recovery_2d8e5e.py` (Folder 17)
- `test_samplers_341888.py` (Folder 17)
- `test_sensor_7c7695.py` (Folder 17)
- `test_simple_source_67978c.py` (Folder 17)
- `test_sort_index_efa5b2.py` (Folder 17)
- `test_sort_values_bf73ae.py` (Folder 17)
- `test_transformation_24b2f2.py` (Folder 17)
- `test_trino_ecb53b.py` (Folder 17)
- `test_union_categoricals_f7953f.py` (Folder 17)
- `test_update_2ec5bf.py` (Folder 17)
- `test_validators_c14cb1.py` (Folder 17)
- `test_zrouting_59f6cb.py` (Folder 17)
- `timeout_ad63a6.py` (Folder 17)
- `timeout_e854c4.py` (Folder 17)
- `tokenize_4942d9.py` (Folder 17)
- `translate_e86f1f.py` (Folder 17)
- `type_cameras_3d7c68.py` (Folder 17)
- `type_var_a00119.py` (Folder 17)
- `ui_89cc2d.py` (Folder 17)
- `update_dd0d23.py` (Folder 17)
- `utils_a09b8c.py` (Folder 17)
- `utils_dae3e4.py` (Folder 17)
- `wallets_eeffd4.py` (Folder 17)
- `web_ws_e88d1e.py` (Folder 17)
- `worker_49ac2b.py` (Folder 17)
- `conftest_dc3d22.py` (Folder 18)
- `device_tracker_6dfc88.py` (Folder 18)
- `instantiate_c1da76.py` (Folder 18)
- `test_file_buffer_url_2ebfea.py` (Folder 18)
- `test_rpc_apiserver_00e9ed.py` (Folder 18)

---

## Files with Changes (76 total)

### _misc_3826f1.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 3 → 4 **(+1 added)**

### monitor_7b218d.py

**Changes detected**: method_removed_in_typed

#### What Changed:

- ❌ **Method removed** from `Monitor`: `on_assignment_error`
- ❌ **Method removed** from `Monitor`: `on_send_error`
- ❌ **Method removed** from `Monitor`: `_normalize`
- ❌ **Method removed** from `Monitor`: `on_assignment_start`
- ❌ **Method removed** from `Monitor`: `on_rebalance_end`
- ❌ **Method removed** from `Monitor`: `on_rebalance_start`
- ❌ **Method removed** from `Monitor`: `on_tp_commit`
- ❌ **Method removed** from `Monitor`: `on_commit_completed`
- ❌ **Method removed** from `Monitor`: `on_web_request_start`
- ❌ **Method removed** from `Monitor`: `on_commit_initiated`
- ❌ **Method removed** from `Monitor`: `on_rebalance_return`
- ❌ **Method removed** from `Monitor`: `on_assignment_completed`
- ❌ **Method removed** from `Monitor`: `on_send_completed`
- ❌ **Method removed** from `Monitor`: `on_send_initiated`
- ❌ **Method removed** from `Monitor`: `track_tp_end_offset`
- ❌ **Method removed** from `Monitor`: `count`
- ❌ **Method removed** from `Monitor`: `on_web_request_end`

### test_monitor_31e01e.py

**Changes detected**: class_added_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ✅ **Class added**: `RebalanceState`
- ✅ **Class added**: `SendState`
- ✅ **Class added**: `WebRequestState`
- ✅ **Class added**: `AssignmentState`
- ✅ **Class added**: `EventState`
- ❌ **Method removed** from `test_Monitor`: `test__sample`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 1 → 0 **(-1 removed)**

### trainer_test_f0df2a.py

**Changes detected**: class_missing_in_typed, method_added_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `RecordMetricLearningRateScheduler`
- ❌ **Class removed**: `FakeTrainerCallback`
- ❌ **Class removed**: `SlowDataLoader`
- ❌ **Class removed**: `TestAmpTrainer`
- ❌ **Class removed**: `FakeOnBatchCallback`
- ❌ **Class removed**: `TestSparseClipGrad`
- ✅ **Method added** to `FakeModel`: `__init__`
- ⚙️ **Function signature changed**: `__init__`
  - Original params: `self, optimizer`
  - Modified params: `self, vocab`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 11 → 2 **(-9 removed)**
  - `for_loops`: 24 → 8 **(-16 removed)**
  - `with_statements`: 6 → 2 **(-4 removed)**

### api_test_5baf5c.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `with_statements`: 5 → 2 **(-3 removed)**

### channel_43e477.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 166 → 38 **(-128 removed)**
  - `for_loops`: 8 → 0 **(-8 removed)**
  - `try_except`: 2 → 1 **(-1 removed)**

### dataclasses_a27d2a.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `__init__`
  - Original params: `self`
  - Modified params: `self, dc_cls`

### ewm_ef0fa0.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `ExponentialMovingWindowGroupby`
- ❌ **Class removed**: `OnlineExponentialMovingWindow`
- ❌ **Method removed** from `ExponentialMovingWindow`: `corr`
- ⚙️ **Function signature changed**: `__init__`
  - Original params: `self, obj, com, span, halflife, alpha, min_periods, adjust, ignore_na, times, engine, engine_kwargs, selection`
  - Modified params: `self, obj, com, span, halflife, alpha, min_periods, adjust, ignore_na, times, method, selection`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 43 → 33 **(-10 removed)**
  - `with_statements`: 1 → 0 **(-1 removed)**

### strings_8fc975.py

**Changes detected**: method_removed_in_typed

#### What Changed:

- ❌ **Method removed** from `StringMethods`: `get_dummies`
- ❌ **Method removed** from `StringMethods`: `zfill`

### test_planner_aec317.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestPlanRestAPI`
- ❌ **Class removed**: `TestKeyVariable`
- ❌ **Class removed**: `TestPlanWebsocketAPI`
- ❌ **Class removed**: `TestPlanCreateUpdateAPIMapping`
- ❌ **Class removed**: `TestUnreferencedResourcePlanner`
- ❌ **Class removed**: `TestPlanLogGroup`
- ❌ **Class removed**: `Foo`
- ❌ **Class removed**: `TestRemoteState`
- ❌ **Class removed**: `TestPlanSNSSubscription`
- ❌ **Class removed**: `TestPlanManagedRole`
- ❌ **Class removed**: `TestPlanLambdaFunction`
- ❌ **Class removed**: `TestPlanScheduledEvent`
- ❌ **Class removed**: `TestPlanDynamoDBSubscription`
- ❌ **Class removed**: `TestPlanCloudWatchEvent`
- ❌ **Class removed**: `TestPlanCreateUpdateDomainName`
- ❌ **Class removed**: `TestPlanS3Events`
- ❌ **Class removed**: `TestPlanSQSSubscription`
- ❌ **Class removed**: `TestPlanKinesisSubscription`
- ⚙️ **Function signature changed**: `create_api_mapping`
  - Original params: `self`
  - Modified params: ``
- 🔄 **Control flow structure modified**:
  - `if_statements`: 14 → 12 **(-2 removed)**
  - `with_statements`: 4 → 0 **(-4 removed)**

### client_aeeaee.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 23 → 20 **(-3 removed)**
  - `try_except`: 14 → 12 **(-2 removed)**

### test_btanalysis_4aed6c.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `with_statements`: 20 → 14 **(-6 removed)**

### test_counts_dbbc0a.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestCountStats`
- ❌ **Class removed**: `TestLoggingCountStats`
- ❌ **Class removed**: `TestActiveUsersAudit`
- ❌ **Class removed**: `TestRealmActiveHumans`
- ❌ **Class removed**: `GetLastIdFromServerTest`
- ❌ **Class removed**: `TestDoIncrementLoggingStat`
- ❌ **Class removed**: `TestDoAggregateToSummaryTable`
- ❌ **Class removed**: `TestDeleteStats`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 8 → 5 **(-3 removed)**
  - `for_loops`: 26 → 9 **(-17 removed)**
  - `with_statements`: 14 → 4 **(-10 removed)**

### cache_83a043.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `IgnoreUnhashableLruCacheWrapper`
- ⚙️ **Function signature changed**: `decorator`
  - Original params: `user_function`
  - Modified params: `func`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 29 → 20 **(-9 removed)**
  - `try_except`: 7 → 5 **(-2 removed)**

### data_catalog_8d7fcf.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 19 → 20 **(+1 added)**

### users_e38fbd.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 32 → 22 **(-10 removed)**
  - `for_loops`: 16 → 13 **(-3 removed)**
  - `try_except`: 1 → 0 **(-1 removed)**

### import_export_tests_dc500f.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `TestImportExport`: `test_import_table_1_col_1_met`
- ❌ **Method removed** from `TestImportExport`: `test_import_override_dashboard_slice_reset_ownership`
- ❌ **Method removed** from `TestImportExport`: `test_import_table_no_metadata`
- ❌ **Method removed** from `TestImportExport`: `test_import_override_dashboard_2_slices`
- ❌ **Method removed** from `TestImportExport`: `test_import_table_override_identical`
- ❌ **Method removed** from `TestImportExport`: `test_import_table_2_col_2_met`
- ❌ **Method removed** from `TestImportExport`: `_create_dashboard_for_import`
- ❌ **Method removed** from `TestImportExport`: `test_import_table_override`
- ❌ **Method removed** from `TestImportExport`: `test_import_new_dashboard_slice_reset_ownership`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 10 → 9 **(-1 removed)**

### test_period_c61bbd.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestPeriodMethods`
- ❌ **Class removed**: `TestPeriodProperties`
- ❌ **Class removed**: `TestPeriodComparisons`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_weekly`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_mult`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_large_ordinal`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_quarterly`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_constructor_nanosecond`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_deprecated_lowercase_freq`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_nat`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_construct_from_nat_string_and_freq`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_parse_week_str_roundstrip`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_annual`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_from_ordinal`
- ❌ **Method removed** from `TestPeriodConstruction`: `test_period_cons_combined`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 5 → 0 **(-5 removed)**
  - `for_loops`: 13 → 0 **(-13 removed)**
  - `with_statements`: 51 → 26 **(-25 removed)**

### test_runner_4222e5.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestPrefectDbtRunnerEvents`
- ❌ **Class removed**: `TestPrefectDbtRunnerLineage`
- ❌ **Class removed**: `TestPrefectDbtRunnerLogging`
- ❌ **Method removed** from `TestPrefectDbtRunnerInvoke`: `test_invoke_with_manifest_requiring_commands`
- ❌ **Method removed** from `TestPrefectDbtRunnerInvoke`: `test_invoke_with_preloaded_manifest`
- ❌ **Method removed** from `TestPrefectDbtRunnerInvoke`: `test_failure_result_types`
- ❌ **Method removed** from `TestPrefectDbtRunnerInvoke`: `test_invoke_debug_command`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 47 → 16 **(-31 removed)**
  - `for_loops`: 1 → 0 **(-1 removed)**
  - `with_statements`: 15 → 6 **(-9 removed)**

### test_base_indexer_9339e3.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `with_statements`: 13 → 5 **(-8 removed)**

### test_find_replace_bc2ab9.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 23 → 18 **(-5 removed)**
  - `with_statements`: 15 → 5 **(-10 removed)**

### test_numba_246dfa.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `with_statements`: 10 → 9 **(-1 removed)**

### test_payments_dde789.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `with_statements`: 7 → 4 **(-3 removed)**

### test_process_pending_consolidations_a9c6ec.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 7 → 0 **(-7 removed)**
  - `for_loops`: 9 → 3 **(-6 removed)**

### textual_entailment_suite_19763e.py

**Changes detected**: method_removed_in_typed

#### What Changed:

- ❌ **Method removed** from `TextualEntailmentSuite`: `_default_fairness_tests`
- ❌ **Method removed** from `TextualEntailmentSuite`: `_default_temporal_tests`
- ❌ **Method removed** from `TextualEntailmentSuite`: `_default_ner_tests`

### api_tests_b98962.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestGetChartDataApi`
- ❌ **Class removed**: `QueryContext`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_in_op_filter__data_is_returned`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_filter_suppose_to_return_empty_data__no_data_returned`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_row_limit_and_offset__row_limit_and_offset_were_applied`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_virtual_table_with_colons_as_datasource`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_invalid_where_parameter__400`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_async`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_orderby_parameter_with_second_query__400`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_not_permitted_actor__403`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_applied_time_extras`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_invalid_datasource__400`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_annotations_layers__annotations_data_returned`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_invalid_post_processing`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_prophet`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_async_invalid_token`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_adhoc_column_without_metrics`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_invalid_having_parameter_closing_and_comment__400`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_async_cached_sync_response`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_dttm_filter`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_where_parameter_including_comment___200`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_when_where_parameter_is_template_and_query_result_type__query_is_templated`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_timegrains_and_columns_result_types`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_table_columns_without_metrics`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_invalid_where_parameter_closing_unclosed__400`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_query_result_type_and_non_existent_filter__filter_omitted`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_rowcount`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_with_series_limit`
- ❌ **Method removed** from `TestPostChartDataApi`: `test_chart_data_async_results_type`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 16 → 3 **(-13 removed)**
  - `for_loops`: 1 → 0 **(-1 removed)**
  - `with_statements`: 4 → 1 **(-3 removed)**

### test_c_parser_only_bf97b6.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `NoNextBuffer`
- 🔄 **Control flow structure modified**:
  - `with_statements`: 18 → 7 **(-11 removed)**

### test_categorical_c93963.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 73 → 6 **(-67 removed)**
  - `for_loops`: 5 → 1 **(-4 removed)**
  - `with_statements`: 10 → 0 **(-10 removed)**

### test_parse_dates_206fdc.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 10 → 7 **(-3 removed)**
  - `with_statements`: 10 → 4 **(-6 removed)**

### test_setitem_31620d.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `CoercionTest`
- ❌ **Class removed**: `TestSetitemDT64IntoInt`
- ❌ **Class removed**: `SetitemCastingEquivalents`
- ❌ **Class removed**: `TestSetitemNAPeriodDtype`
- ❌ **Class removed**: `TestSetitemNADatetimeLikeDtype`
- ❌ **Class removed**: `TestCoercionInt8`
- ❌ **Class removed**: `TestCoercionDatetime64TZ`
- ❌ **Class removed**: `TestCoercionDatetime64HigherReso`
- ❌ **Class removed**: `TestSmallIntegerSetitemUpcast`
- ❌ **Class removed**: `TestCoercionComplex`
- ❌ **Class removed**: `TestSetitemTimedelta64IntoNumeric`
- ❌ **Class removed**: `TestSetitemCastingEquivalents`
- ❌ **Class removed**: `TestCoercionObject`
- ❌ **Class removed**: `TestSetitemWithExpansion`
- ❌ **Class removed**: `TestSetitemRangeIntoIntegerSeries`
- ❌ **Class removed**: `TestSetitemCallable`
- ❌ **Class removed**: `TestSetitemMismatchedTZCastsToObject`
- ❌ **Class removed**: `TestCoercionBool`
- ❌ **Class removed**: `TestCoercionFloat32`
- ❌ **Class removed**: `TestSetitemFloatNDarrayIntoIntegerSeries`
- ❌ **Class removed**: `TestPeriodIntervalCoercion`
- ❌ **Class removed**: `TestSetitemCasting`
- ❌ **Class removed**: `TestCoercionInt64`
- ❌ **Class removed**: `TestCoercionFloat64`
- ❌ **Class removed**: `TestCoercionDatetime64`
- ❌ **Class removed**: `TestSetitemIntoIntegerSeriesNeedsUpcast`
- ❌ **Class removed**: `TestCoercionTimedelta64`
- ❌ **Class removed**: `TestCoercionString`
- ❌ **Class removed**: `TestSeriesNoneCoercion`
- ❌ **Class removed**: `TestSetitemFloatIntervalWithIntIntervalValues`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 30 → 0 **(-30 removed)**
  - `for_loops`: 2 → 0 **(-2 removed)**
  - `with_statements`: 51 → 9 **(-42 removed)**

### transformation_82baef.py

**Changes detected**: parameter_mismatch, control_flow_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `p`
  - Original params: `dataset`
  - Modified params: `new_df`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 19 → 12 **(-7 removed)**
  - `for_loops`: 1 → 0 **(-1 removed)**

### web_rtc_831f2f.py

**Changes detected**: method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `WebRTCManager`: `_process_signaling_message`
- ❌ **Method removed** from `WebRTCManager`: `_set_candidates_for_address`
- ❌ **Method removed** from `WebRTCManager`: `close_connection`
- ❌ **Method removed** from `WebRTCManager`: `stop`
- ❌ **Method removed** from `WebRTCManager`: `process_signaling_message`
- ❌ **Method removed** from `WebRTCManager`: `_reset_state`
- ❌ **Method removed** from `WebRTCManager`: `send_message`
- ❌ **Method removed** from `WebRTCManager`: `has_ready_channel`
- ❌ **Method removed** from `WebRTCManager`: `_add_connection`
- ❌ **Method removed** from `WebRTCManager`: `_wrapped_initialize_web_rtc`
- ❌ **Method removed** from `WebRTCManager`: `_initialize_web_rtc`
- ❌ **Method removed** from `WebRTCManager`: `get_channel_init_timeout`
- ❌ **Method removed** from `WebRTCManager`: `_handle_ice_connection_closed`
- ❌ **Method removed** from `WebRTCManager`: `health_check`
- ❌ **Method removed** from `WebRTCManager`: `_process_signaling_for_address`
- ⚙️ **Function signature changed**: `send_message`
  - Original params: `self, partner_address, message`
  - Modified params: `self, message`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 38 → 20 **(-18 removed)**
  - `for_loops`: 7 → 5 **(-2 removed)**
  - `while_loops`: 1 → 0 **(-1 removed)**
  - `with_statements`: 2 → 0 **(-2 removed)**

### test_integration_ae2f32.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 5 → 0 **(-5 removed)**

### auth_d8f17e.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `GoogleOAuth2Mixin`
- ❌ **Class removed**: `FacebookGraphMixin`
- ❌ **Method removed** from `TwitterMixin`: `_oauth_consumer_token`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 59 → 50 **(-9 removed)**
  - `for_loops`: 4 → 3 **(-1 removed)**

### entity_0d0e17.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 33 → 34 **(+1 added)**

### metrics_b562d7.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 9 → 11 **(+2 added)**

### superset_factory_util_7e174c.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `insert_model`
  - Original params: `dashboard`
  - Modified params: `model`

### test_generic_7cde9d.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `TestNDFrame`: `test_flags_identity`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 9 → 8 **(-1 removed)**

### test_init_f7cb07.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `MockLightEntityEntity`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 6 → 0 **(-6 removed)**
  - `for_loops`: 4 → 0 **(-4 removed)**
  - `with_statements`: 14 → 0 **(-14 removed)**

### test_process_sync_aggregate_c588af.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 3 → 0 **(-3 removed)**
  - `for_loops`: 7 → 2 **(-5 removed)**

### test_session_extension_hooks_7b3b93.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestKedroContextSpecsHook`
- ❌ **Class removed**: `TestAsyncNodeDatasetHooks`
- ❌ **Class removed**: `LogCatalog`
- ❌ **Method removed** from `TestBeforeNodeRunHookWithInputUpdates`: `test_broken_input_update_parallel`
- ❌ **Method removed** from `TestBeforeNodeRunHookWithInputUpdates`: `test_broken_input_update`
- ❌ **Method removed** from `TestBeforeNodeRunHookWithInputUpdates`: `test_correct_input_update_parallel`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 1 → 0 **(-1 removed)**
  - `with_statements`: 5 → 3 **(-2 removed)**

### objects_2ea8dd.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `get_window_bounds`
  - Original params: `self, num_values, min_periods, center, closed, step`
  - Modified params: `self, num_values, min_periods, center, closed, step, win_type`

### test_datetime64_b94434.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestDatetime64DateOffsetArithmetic`
- ❌ **Class removed**: `TestDatetime64OverflowHandling`
- ❌ **Class removed**: `TestDatetime64Arithmetic`
- ❌ **Class removed**: `TestDatetimeIndexArithmetic`
- ❌ **Class removed**: `TestTimestampSeriesArithmetic`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_scalar_comparison_tzawareness`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_comparison_tzawareness_compat_scalars`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_dti_cmp_str`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_comparison_tzawareness_compat`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_dti_cmp_nat_behaves_like_float_cmp_nan`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_nat_comparison_tzawareness`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_dti_cmp_list`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_dti_cmp_tdi_tzawareness`
- ❌ **Method removed** from `TestDatetimeIndexComparisons`: `test_dti_cmp_object_dtype`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 26 → 11 **(-15 removed)**
  - `for_loops`: 14 → 0 **(-14 removed)**
  - `with_statements`: 79 → 8 **(-71 removed)**

### test_pythonapi_d455c1.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 1 → 0 **(-1 removed)**
  - `for_loops`: 2 → 1 **(-1 removed)**
  - `with_statements`: 20 → 12 **(-8 removed)**

### test_quantile_94afc3.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestQuantileExtensionDtype`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_item_cache`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_empty_no_rows_dt64`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_empty_no_rows_ints`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_invalid`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_nan`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_table_invalid_interpolation`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_nat`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_empty_no_columns`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_empty_no_rows_floats`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_box_nat`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_quantile_box`
- ❌ **Method removed** from `TestDataFrameQuantile`: `test_invalid_method`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 19 → 12 **(-7 removed)**
  - `with_statements`: 6 → 3 **(-3 removed)**

### test_range_fc6fe9.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `TestRangeIndex`: `test_len_specialised`
- ❌ **Method removed** from `TestRangeIndex`: `test_append_len_one`
- ❌ **Method removed** from `TestRangeIndex`: `test_sort_values_key`
- ❌ **Method removed** from `TestRangeIndex`: `test_isin_range`
- ❌ **Method removed** from `TestRangeIndex`: `test_range_index_rsub_by_const`
- ❌ **Method removed** from `TestRangeIndex`: `test_engineless_lookup`
- ❌ **Method removed** from `TestRangeIndex`: `test_append`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 3 → 1 **(-2 removed)**
  - `with_statements`: 9 → 5 **(-4 removed)**

### test_common_27891c.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestError`
- ❌ **Method removed** from `TestMMapWrapper`: `test_unknown_engine`
- ❌ **Method removed** from `TestMMapWrapper`: `test_binary_mode`
- ❌ **Method removed** from `TestMMapWrapper`: `test_next`
- ❌ **Method removed** from `TestMMapWrapper`: `test_warning_missing_utf_bom`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 9 → 6 **(-3 removed)**
  - `for_loops`: 1 → 0 **(-1 removed)**
  - `with_statements`: 50 → 16 **(-34 removed)**

### test_iloc_211b3a.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestILocErrors`
- ❌ **Class removed**: `TO`
- ❌ **Class removed**: `TestILocCallable`
- ❌ **Class removed**: `TestILocSeries`
- ❌ **Class removed**: `TestILocSetItemDuplicateColumns`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_series`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_frame`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_dictionary_value`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_pure_position_based`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_dups`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_slice_negative_step_ea_block`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_with_boolean_operation`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_categorical_values`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_with_duplicates`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_list_of_lists`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_identity_slice_returns_new_object`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_read_only_values`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_mask`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_doc_issue`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_bool_indexer`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_assign_series_to_df_cell`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_int_single_ea_block_view`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_non_unique_indexing`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_categorical_updates_inplace`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_2d_ndarray_into_ea_block`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_multicolumn_to_datetime`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_frame_duplicate_columns_multiple_blocks`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_loc_setitem_boolean_list`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_setitem_mix_of_nan_and_interval`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_readonly_key`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_with_scalar_index`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_interval`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_pandas_object`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_empty_list_indexer_is_ok`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_with_duplicates2`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_series_indexing_zerodim_np_array`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_float_duplicates`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_indexing_zerodim_np_array`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_getitem_labelled_frame`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_list`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_empty_frame_raises_with_3d_ndarray`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_td64_values_cast_na`
- ❌ **Method removed** from `TestiLocBaseIndependent`: `test_iloc_setitem_custom_object`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 8 → 3 **(-5 removed)**
  - `for_loops`: 3 → 0 **(-3 removed)**
  - `try_except`: 1 → 0 **(-1 removed)**
  - `with_statements`: 31 → 15 **(-16 removed)**

### test_plugins_ab0c47.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `Model1`
- ❌ **Class removed**: `Model2`
- ⚙️ **Function signature changed**: `foo`
  - Original params: ``
  - Modified params: `a`
- 🔄 **Control flow structure modified**:
  - `with_statements`: 18 → 16 **(-2 removed)**

### test_replace_ac1a27.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestDataFrameReplaceRegex`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_dict_strings_vs_ints`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_datetimetz`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_ea_ignore_float`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_regex_replace_string_types`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_dtypes`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_limit`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_mixed_int_block_splitting`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_joint_simple_replace_and_regex_replace`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_mixed2`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_NA_with_None`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_dict_category_type`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_list`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_simple_nested_dict_with_nonexistent_value`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_mixed`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_dict_tuple_list_ordering_remains_the_same`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_categorical_replace_with_dict`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_commutative`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_simple_nested_dict`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_series_no_regex`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_nested_dict_overlapping_keys_replace_int`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_invalid_to_replace`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_nullable_int_with_string_doesnt_cast`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_nullable_column`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_list_with_mixed_type`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_None_keeps_categorical`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_no_replacement_dtypes`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_series_dict`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_swapping_bug`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_dict_no_regex`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_bool_with_bool`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_empty_list`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_mixed_int_block_upcasting`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_duplicate_columns`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_empty_dictlike`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_doesnt_replace_without_regex`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_nested_dict_overlapping_keys_replace_str`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_value_is_none`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_mixed3`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_nil_na`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_intervals`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_regex_metachar`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_after_convert_dtypes`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_all_NA`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_unicode`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_replacer_dtype`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_input_formats_listlike`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_bool_with_string`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_truthy`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_dict_with_bool_keys`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_convert`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_input_formats_scalar`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_value_category_type`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_NAT_with_None`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_pure_bool_with_string_no_op`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_for_new_dtypes`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_value_none_dtype_numeric`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_with_compiled_regex`
- ❌ **Method removed** from `TestDataFrameReplace`: `test_replace_bytes`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 9 → 2 **(-7 removed)**
  - `for_loops`: 2 → 0 **(-2 removed)**
  - `with_statements`: 6 → 0 **(-6 removed)**

### btanalysis_7decb4.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 45 → 39 **(-6 removed)**

### test_arithmetic_e20900.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestFrameArithmetic`
- ❌ **Class removed**: `TestFrameArithmeticUnsorted`
- ❌ **Class removed**: `SubclassedSeries`
- ❌ **Class removed**: `TestFrameFlexArithmetic`
- ❌ **Class removed**: `SubclassedDataFrame`
- ❌ **Method removed** from `TestFrameFlexComparisons`: `test_df_flex_cmp_ea_dtype_with_ndarray_series`
- ⚙️ **Function signature changed**: `__init__`
  - Original params: `self, my_extra_data`
  - Modified params: `self, value, dtype`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 21 → 2 **(-19 removed)**
  - `for_loops`: 7 → 0 **(-7 removed)**
  - `with_statements`: 53 → 17 **(-36 removed)**

### test_melt_711512.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestWideToLong`
- ❌ **Class removed**: `TestLreshape`
- ❌ **Method removed** from `TestMelt`: `test_melt_multiindex_columns_var_name_too_many`
- ❌ **Method removed** from `TestMelt`: `test_melt_multiindex_columns_var_name`
- ❌ **Method removed** from `TestMelt`: `test_melt_non_scalar_var_name_raises`
- ❌ **Method removed** from `TestMelt`: `test_melt_allows_non_scalar_id_vars`
- ❌ **Method removed** from `TestMelt`: `test_melt_allows_non_string_var_name`
- ❌ **Method removed** from `TestMelt`: `test_melt_preserves_datetime`
- ❌ **Method removed** from `TestMelt`: `test_melt_ea_columns`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 1 → 0 **(-1 removed)**
  - `with_statements`: 12 → 6 **(-6 removed)**

### test_transform_0bcea9.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 56 → 6 **(-50 removed)**
  - `for_loops`: 12 → 4 **(-8 removed)**
  - `with_statements`: 17 → 7 **(-10 removed)**

### datetime_ba4330.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 20 → 21 **(+1 added)**

### datetimelike_f48bd1.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `insert`
- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `delete`
- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `take`
- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `_get_delete_freq`
- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `_get_insert_freq`
- ❌ **Method removed** from `DatetimeTimedeltaMixin`: `_from_join_target`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 60 → 45 **(-15 removed)**

### mccabe_5703d2.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 20 → 21 **(+1 added)**

### test_blackouts_53e928.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `Blackout`
- ❌ **Class removed**: `CustomNotify`
- ❌ **Method removed** from `BlackoutsTestCase`: `test_edit_blackout`
- ❌ **Method removed** from `BlackoutsTestCase`: `test_combination_blackout`
- ❌ **Method removed** from `BlackoutsTestCase`: `test_user_info`
- ❌ **Method removed** from `BlackoutsTestCase`: `test_custom_notify`
- ❌ **Method removed** from `BlackoutsTestCase`: `test_origin_blackout`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 5 → 0 **(-5 removed)**

### test_data_io_365c19.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 12 → 4 **(-8 removed)**
  - `for_loops`: 10 → 4 **(-6 removed)**
  - `while_loops`: 6 → 0 **(-6 removed)**
  - `with_statements`: 13 → 6 **(-7 removed)**

### test_microbatch_124d73.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestMicrobatchModelStoppedByKeyboardInterrupt`
- ❌ **Class removed**: `TestMicrobatchJinjaContext`
- ❌ **Class removed**: `TestCanSilenceInvalidConcurrentBatchesConfigWarning`
- ❌ **Class removed**: `TestMicrobatchUsingRefRenderSkipsFilter`
- ❌ **Class removed**: `TestMicrobatchJinjaContextVarsAvailable`
- ❌ **Class removed**: `TestMicrobatchIncrementalBatchFailure`
- ❌ **Class removed**: `TestMicrobatchModelSkipped`
- ❌ **Class removed**: `TestMicrobatchInitialBatchFailure`
- ❌ **Class removed**: `TestWhenOnlyOneBatchRunBothPostAndPreHooks`
- ❌ **Class removed**: `TestMicrbobatchModelsRunWithSameCurrentTime`
- ❌ **Class removed**: `TestMicrobatchFullRefreshConfigFalse`
- ❌ **Class removed**: `TestMicrobatchRetriesPartialSuccesses`
- ❌ **Class removed**: `TestMicrobatchSecondBatchFailure`
- ❌ **Class removed**: `TestMicrobatchWithInputWithoutEventTime`
- ❌ **Class removed**: `TestMicrobatchCompiledRunPaths`
- ❌ **Class removed**: `TestFirstAndLastBatchAlwaysSequential`
- ❌ **Class removed**: `TestMicrobatchMultipleRetries`
- ❌ **Class removed**: `TestMicrobatchCanRunParallelOrSequential`
- ❌ **Class removed**: `TestFirstBatchRunsPreHookLastBatchRunsPostHook`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 6 → 2 **(-4 removed)**
  - `for_loops`: 6 → 1 **(-5 removed)**
  - `try_except`: 1 → 0 **(-1 removed)**
  - `with_statements`: 45 → 13 **(-32 removed)**

### test_shrinker_1f54e0.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `BadShrinker`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 47 → 34 **(-13 removed)**
  - `for_loops`: 11 → 8 **(-3 removed)**
  - `with_statements`: 1 → 0 **(-1 removed)**

### accessors_09b267.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `CachedSparkFrameMethods`
- ❌ **Method removed** from `SparkFrameMethods`: `analyzed`
- ❌ **Method removed** from `SparkFrameMethods`: `local_checkpoint`
- ❌ **Method removed** from `SparkFrameMethods`: `checkpoint`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 20 → 19 **(-1 removed)**

### fan_63b94a.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `XiaomiAirFresh`
- ❌ **Class removed**: `XiaomiFanMiot`
- ❌ **Class removed**: `XiaomiFanP5`
- ❌ **Class removed**: `XiaomiFan`
- ❌ **Class removed**: `XiaomiAirPurifierMiot`
- ❌ **Class removed**: `XiaomiAirFreshA1`
- ❌ **Class removed**: `XiaomiAirFreshT2017`
- ❌ **Class removed**: `XiaomiFan1C`
- ❌ **Class removed**: `XiaomiAirPurifierMB4`
- ❌ **Class removed**: `XiaomiGenericFan`
- ❌ **Class removed**: `XiaomiFanZA5`
- ❌ **Method removed** from `XiaomiAirPurifier`: `percentage`
- ❌ **Method removed** from `XiaomiAirPurifier`: `operation_mode_class`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 86 → 23 **(-63 removed)**

### sqla_models_tests_7b5225.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `TestDatabaseModel`: `test_labels_expected_on_mutated_query`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 5 → 3 **(-2 removed)**
  - `for_loops`: 6 → 4 **(-2 removed)**
  - `with_statements`: 8 → 5 **(-3 removed)**

### test_init_75a7d5.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `MockLogbookPlatform`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 8 → 0 **(-8 removed)**
  - `for_loops`: 3 → 0 **(-3 removed)**
  - `with_statements`: 3 → 1 **(-2 removed)**

### __init___3f6268.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `Stream`: `get_diagnostics`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 31 → 26 **(-5 removed)**

### aiokafka_d62f1d.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `Transport`
- ❌ **Class removed**: `Producer`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_log_slow_processing`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_ensure_consumer`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `seek`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_log_slow_processing_stream`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `highwater`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `topic_partitions`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `key_partition`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_verify_aiokafka_event_path`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `verify_event_path`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `verify_recovery_event_path`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_make_slow_processing_error`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `_log_slow_processing_commit`
- ❌ **Method removed** from `AIOKafkaConsumerThread`: `assignment`
- ⚙️ **Function signature changed**: `__init__`
  - Original params: `self`
  - Modified params: `self, thread`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 74 → 13 **(-61 removed)**
  - `for_loops`: 2 → 0 **(-2 removed)**
  - `try_except`: 6 → 2 **(-4 removed)**
  - `with_statements`: 1 → 0 **(-1 removed)**

### rewards_039ecf.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 40 → 36 **(-4 removed)**
  - `for_loops`: 18 → 11 **(-7 removed)**

### splitters_9b6f18.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 13 → 2 **(-11 removed)**

### streams_d2f3a4.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Method removed** from `DataQueue`: `__aiter__`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 58 → 54 **(-4 removed)**
  - `try_except`: 7 → 6 **(-1 removed)**

### task_engine_88162b.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `AsyncTaskRunEngine`
- ❌ **Method removed** from `SyncTaskRunEngine`: `transaction_context`
- ❌ **Method removed** from `SyncTaskRunEngine`: `initialize_run`
- ❌ **Method removed** from `SyncTaskRunEngine`: `run_context`
- ❌ **Method removed** from `SyncTaskRunEngine`: `start`
- ❌ **Method removed** from `SyncTaskRunEngine`: `handle_crash`
- ❌ **Method removed** from `SyncTaskRunEngine`: `call_task_fn`
- ❌ **Method removed** from `SyncTaskRunEngine`: `setup_run_context`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 105 → 45 **(-60 removed)**
  - `for_loops`: 3 → 2 **(-1 removed)**
  - `while_loops`: 8 → 1 **(-7 removed)**
  - `try_except`: 18 → 6 **(-12 removed)**
  - `with_statements`: 19 → 0 **(-19 removed)**

### sql_parse_tests_ebc947.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 4 → 0 **(-4 removed)**
  - `for_loops`: 2 → 0 **(-2 removed)**
  - `with_statements`: 9 → 0 **(-9 removed)**

### test_fillna_057fac.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestFillnaPad`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_period`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_method_and_limit_invalid`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_f32_upcast_with_dict`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_dt64_non_nao`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_categorical`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_dt64_timestamp`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_categorical_raises`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_listlike_invalid`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_pytimedelta`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_categorical_accept_same_type`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_float_casting`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_categorical_with_new_categories`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_dt64tz_with_method`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_numeric_inplace`
- ❌ **Method removed** from `TestSeriesFillNA`: `test_fillna_datetime64_with_timezone_tzinfo`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 3 → 2 **(-1 removed)**
  - `for_loops`: 2 → 1 **(-1 removed)**
  - `with_statements`: 10 → 1 **(-9 removed)**

### test_hashtable_89c8a4.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestHelpFunctions`
- ❌ **Class removed**: `TestHelpFunctionsWithNans`
- ❌ **Class removed**: `TestHashTableWithNans`
- ❌ **Method removed** from `TestPyObjectHashTableWithNans`: `test_nan_in_nested_namedtuple`
- ⚙️ **Function signature changed**: `test_map_locations`
  - Original params: `self, table_type, dtype`
  - Modified params: `self, table_type, dtype, writable`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 10 → 8 **(-2 removed)**
  - `with_statements`: 15 → 12 **(-3 removed)**

### test_purge_979a37.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 15 → 0 **(-15 removed)**
  - `for_loops`: 37 → 3 **(-34 removed)**
  - `with_statements`: 88 → 23 **(-65 removed)**

### test_readers_6fbe49.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- ❌ **Class removed**: `TestExcelFileRead`
- ❌ **Method removed** from `TestReaders`: `test_reading_all_sheets`
- ❌ **Method removed** from `TestReaders`: `test_reader_seconds`
- ❌ **Method removed** from `TestReaders`: `test_bad_sheetname_raises`
- ❌ **Method removed** from `TestReaders`: `test_ignore_chartsheets_by_str`
- ❌ **Method removed** from `TestReaders`: `test_missing_file_raises`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_nrows`
- ❌ **Method removed** from `TestReaders`: `test_close_from_py_localpath`
- ❌ **Method removed** from `TestReaders`: `test_read_from_http_url`
- ❌ **Method removed** from `TestReaders`: `test_dtype_backend_string`
- ❌ **Method removed** from `TestReaders`: `test_bad_engine_raises`
- ❌ **Method removed** from `TestReaders`: `test_read_from_s3_url`
- ❌ **Method removed** from `TestReaders`: `test_excel_read_buffer`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_multiindex_blank_after_name`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_ods_nested_xml`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_multiindex_header_only`
- ❌ **Method removed** from `TestReaders`: `test_reader_converters`
- ❌ **Method removed** from `TestReaders`: `test_corrupt_bytes_raises`
- ❌ **Method removed** from `TestReaders`: `test_excel_old_index_format`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_nrows_greater_than_nrows_in_file`
- ❌ **Method removed** from `TestReaders`: `test_read_from_pathlib_path`
- ❌ **Method removed** from `TestReaders`: `test_date_conversion_overflow`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_skiprows_callable_not_in`
- ❌ **Method removed** from `TestReaders`: `test_trailing_blanks`
- ❌ **Method removed** from `TestReaders`: `test_read_from_s3_object`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_nrows_params`
- ❌ **Method removed** from `TestReaders`: `test_exception_message_includes_sheet_name`
- ❌ **Method removed** from `TestReaders`: `test_dtype_mangle_dup_cols`
- ❌ **Method removed** from `TestReaders`: `test_reader_dtype`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_blank_with_header`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_bool_header_arg`
- ❌ **Method removed** from `TestReaders`: `test_dtype_backend_and_dtype`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_nrows_non_integer_parameter`
- ❌ **Method removed** from `TestReaders`: `test_one_col_noskip_blank_line`
- ❌ **Method removed** from `TestReaders`: `test_no_header_with_list_index_col`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_skiprows`
- ❌ **Method removed** from `TestReaders`: `test_ignore_chartsheets_by_int`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_blank`
- ❌ **Method removed** from `TestReaders`: `test_deprecated_kwargs`
- ❌ **Method removed** from `TestReaders`: `test_reader_dtype_str`
- ❌ **Method removed** from `TestReaders`: `test_reader_special_dtypes`
- ❌ **Method removed** from `TestReaders`: `test_sheet_name`
- ❌ **Method removed** from `TestReaders`: `test_reading_all_sheets_with_blank`
- ❌ **Method removed** from `TestReaders`: `test_dtype_backend`
- ❌ **Method removed** from `TestReaders`: `test_read_excel_multiindex`
- ❌ **Method removed** from `TestReaders`: `test_euro_decimal_format`
- ❌ **Method removed** from `TestReaders`: `test_reader_spaces`
- ❌ **Method removed** from `TestReaders`: `test_reading_multiple_specific_sheets`
- ❌ **Method removed** from `TestReaders`: `test_read_from_file_url`
- ❌ **Method removed** from `TestReaders`: `test_multiheader_two_blank_lines`
- ⚙️ **Function signature changed**: `test_engine_used`
  - Original params: `self, read_ext, engine`
  - Modified params: `self, read_ext, engine, monkeypatch`
- 🔄 **Control flow structure modified**:
  - `if_statements`: 38 → 15 **(-23 removed)**
  - `for_loops`: 2 → 0 **(-2 removed)**
  - `try_except`: 2 → 0 **(-2 removed)**
  - `with_statements`: 59 → 9 **(-50 removed)**

