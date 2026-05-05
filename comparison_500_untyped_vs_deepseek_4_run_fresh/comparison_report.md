# AST Comparison Report: DeepSeek 4 Run vs Untyped Baseline

## Summary Statistics

- **Total files compared**: 500
- **Files with structural changes**: 70
- **Files with parse errors**: 275
- **Files without baseline**: 0
- **Percentage with changes**: 14.00%
- **Percentage with parse errors**: 55.00%
- **Percentage without baseline**: 0.00%

### Key Insights
- Out of 500 files, **70 (14.00%)** had structural differences
- **275** files could not be parsed (syntax errors in DeepSeek output)
- Most changes involve **control flow modifications**, **class additions**, or **parameter changes**

---

## Warning: Parse Errors - DeepSeek Generated Invalid Syntax

These files could not be analyzed because the generated code has syntax errors:

**_client_5a9e7d.py**

```
Error: unterminated triple-quoted string literal (detected at line 411) (<unknown>, line 391)
```

**_misc_3826f1.py**

```
Error: '[' was never closed (<unknown>, line 403)
```

**alarm_control_panel_e80274.py**

```
Error: '(' was never closed (<unknown>, line 234)
```

**base_6106b5.py**

```
Error: '(' was never closed (<unknown>, line 368)
```

**base_9c14fc.py**

```
Error: unterminated string literal (detected at line 321) (<unknown>, line 321)
```

**channels_8e6ab8.py**

```
Error: '(' was never closed (<unknown>, line 358)
```

**classification_92cb06.py**

```
Error: unterminated triple-quoted string literal (detected at line 321) (<unknown>, line 296)
```

**client_1e3c0a.py**

```
Error: '(' was never closed (<unknown>, line 385)
```

**conftest_ca46fe.py**

```
Error: '(' was never closed (<unknown>, line 356)
```

**elmo_4ed8b3.py**

```
Error: '(' was never closed (<unknown>, line 303)
```

**httpclient_e36adc.py**

```
Error: unterminated triple-quoted string literal (detected at line 355) (<unknown>, line 294)
```

**monitor_7b218d.py**

```
Error: '(' was never closed (<unknown>, line 281)
```

**network_7ec1c6.py**

```
Error: '(' was never closed (<unknown>, line 237)
```

**params_3f7432.py**

```
Error: '(' was never closed (<unknown>, line 308)
```

**test_business_hour_6b0e7b.py**

```
Error: unexpected unindent (<unknown>, line 197)
```

**test_callables_9ff673.py**

```
Error: unterminated string literal (detected at line 251) (<unknown>, line 251)
```

**test_monitor_31e01e.py**

```
Error: '(' was never closed (<unknown>, line 293)
```

**test_value_counts_77b3d1.py**

```
Error: expected ':' (<unknown>, line 201)
```

**train_test_b790e4.py**

```
Error: '{' was never closed (<unknown>, line 249)
```

**_models_dd672f.py**

```
Error: unterminated string literal (detected at line 404) (<unknown>, line 404)
```

**attention_module_7f6559.py**

```
Error: '[' was never closed (<unknown>, line 265)
```

**awsclient_860c67.py**

```
Error: expected ':' (<unknown>, line 354)
```

**base_tests_a1e90e.py**

```
Error: expected an indented block after 'with' statement on line 372 (<unknown>, line 372)
```

**channel_43e477.py**

```
Error: unterminated f-string literal (detected at line 276) (<unknown>, line 276)
```

**ewm_ef0fa0.py**

```
Error: unterminated string literal (detected at line 339) (<unknown>, line 339)
```

**smartcontracts_989d9a.py**

```
Error: unterminated triple-quoted string literal (detected at line 191) (<unknown>, line 191)
```

**strings_8fc975.py**

```
Error: expected ':' (<unknown>, line 604)
```

**test_appgraph_6f35e2.py**

```
Error: unterminated string literal (detected at line 239) (<unknown>, line 239)
```

**test_binance_2f9760.py**

```
Error: unterminated string literal (detected at line 117) (<unknown>, line 117)
```

**test_persistence_02086c.py**

```
Error: unterminated triple-quoted string literal (detected at line 208) (<unknown>, line 173)
```

**test_planner_aec317.py**

```
Error: '(' was never closed (<unknown>, line 237)
```

**testclient_6e1da5.py**

```
Error: '[' was never closed (<unknown>, line 308)
```

**client_aeeaee.py**

```
Error: expected ':' (<unknown>, line 413)
```

**exposed_entities_a550ba.py**

```
Error: '(' was never closed (<unknown>, line 328)
```

**hive_66cde7.py**

```
Error: unexpected unindent (<unknown>, line 315)
```

**imports_c4205d.py**

```
Error: expected ':' (<unknown>, line 354)
```

**methods_fdfe44.py**

```
Error: '[' was never closed (<unknown>, line 299)
```

**routing_66b713.py**

```
Error: unterminated string literal (detected at line 353) (<unknown>, line 353)
```

**test_btanalysis_4aed6c.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**test_conversion_928a39.py**

```
Error: invalid syntax (<unknown>, line 210)
```

**test_counts_dbbc0a.py**

```
Error: '(' was never closed (<unknown>, line 228)
```

**test_dtypes_basic_4a9ebb.py**

```
Error: '[' was never closed (<unknown>, line 233)
```

**test_groupby_86d3bc.py**

```
Error: expected '(' (<unknown>, line 213)
```

**test_nth_cbbcbb.py**

```
Error: '[' was never closed (<unknown>, line 225)
```

**test_plugins_493c2b.py**

```
Error: unterminated string literal (detected at line 274) (<unknown>, line 274)
```

**test_prefect_client_9ba189.py**

```
Error: '(' was never closed (<unknown>, line 377)
```

**test_rank_19ec93.py**

```
Error: '[' was never closed (<unknown>, line 213)
```

**test_reductions_87ff0d.py**

```
Error: invalid syntax (<unknown>, line 229)
```

**test_validator_unittest_4d4a84.py**

```
Error: expected '(' (<unknown>, line 305)
```

**base_9be364.py**

```
Error: unterminated triple-quoted string literal (detected at line 351) (<unknown>, line 338)
```

**cache_83a043.py**

```
Error: expected ':' (<unknown>, line 350)
```

**confluent_3b2473.py**

```
Error: '(' was never closed (<unknown>, line 328)
```

**data_catalog_8d7fcf.py**

```
Error: unterminated triple-quoted string literal (detected at line 422) (<unknown>, line 418)
```

**deposits_09e7ec.py**

```
Error: '(' was never closed (<unknown>, line 261)
```

**regression_5b5987.py**

```
Error: unterminated triple-quoted string literal (detected at line 335) (<unknown>, line 291)
```

**screenshots_949c9a.py**

```
Error: invalid syntax (<unknown>, line 17)
```

**test_ccxt_compat_f96503.py**

```
Error: '(' was never closed (<unknown>, line 291)
```

**test_dtypes_c6ad71.py**

```
Error: unterminated string literal (detected at line 327) (<unknown>, line 327)
```

**test_hist_method_862ac9.py**

```
Error: '(' was never closed (<unknown>, line 263)
```

**test_indexing_e492e0.py**

```
Error: '(' was never closed (<unknown>, line 274)
```

**test_integration_events_f08b97.py**

```
Error: '[' was never closed (<unknown>, line 184)
```

**test_jinja_templated_action_39a881.py**

```
Error: unterminated string literal (detected at line 217) (<unknown>, line 217)
```

**test_join_48601a.py**

```
Error: invalid syntax (<unknown>, line 246)
```

**test_mediatedtransfer_b63130.py**

```
Error: unterminated triple-quoted string literal (detected at line 189) (<unknown>, line 188)
```

**test_read_fwf_ec1bf4.py**

```
Error: unterminated string literal (detected at line 142) (<unknown>, line 142)
```

**test_state_changes_e74ee9.py**

```
Error: '(' was never closed (<unknown>, line 305)
```

**test_utils_4b4e4d.py**

```
Error: '[' was never closed (<unknown>, line 174)
```

**accessors_5aa969.py**

```
Error: unterminated triple-quoted string literal (detected at line 397) (<unknown>, line 395)
```

**color_334856.py**

```
Error: '{' was never closed (<unknown>, line 302)
```

**fbeta_verbose_measure_test_a08d84.py**

```
Error: '(' was never closed (<unknown>, line 270)
```

**test_api_125d2c.py**

```
Error: unterminated string literal (detected at line 256) (<unknown>, line 256)
```

**test_client_1103c5.py**

```
Error: unterminated string literal (detected at line 248) (<unknown>, line 248)
```

**test_filter_rewriting_c801f7.py**

```
Error: expected an indented block after function definition on line 147 (<unknown>, line 148)
```

**test_format_849ee9.py**

```
Error: unterminated string literal (detected at line 292) (<unknown>, line 292)
```

**test_freqai_interface_ca78dd.py**

```
Error: unterminated string literal (detected at line 209) (<unknown>, line 209)
```

**test_hyperopt_ac1e12.py**

```
Error: '[' was never closed (<unknown>, line 188)
```

**test_inference_83c9ac.py**

```
Error: '(' was never closed (<unknown>, line 309)
```

**test_period_c61bbd.py**

```
Error: unterminated string literal (detected at line 297) (<unknown>, line 297)
```

**test_process_withdrawal_request_9f973e.py**

```
Error: invalid syntax (<unknown>, line 245)
```

**test_remote_billing_3a9a00.py**

```
Error: unterminated string literal (detected at line 260) (<unknown>, line 260)
```

**test_runner_4222e5.py**

```
Error: '(' was never closed (<unknown>, line 375)
```

**test_template_ceacd4.py**

```
Error: unterminated string literal (detected at line 275) (<unknown>, line 275)
```

**transport_58bb8a.py**

```
Error: expected ':' (<unknown>, line 329)
```

**celery_tests_fc4a66.py**

```
Error: expected '(' (<unknown>, line 246)
```

**conftest_641a68.py**

```
Error: '(' was never closed (<unknown>, line 303)
```

**dataprovider_a0d211.py**

```
Error: '(' was never closed (<unknown>, line 328)
```

**event_0577a3.py**

```
Error: unterminated triple-quoted string literal (detected at line 371) (<unknown>, line 371)
```

**event_schema_4d75fe.py**

```
Error: unterminated triple-quoted string literal (detected at line 284) (<unknown>, line 281)
```

**manifest_47c52e.py**

```
Error: '(' was never closed (<unknown>, line 370)
```

**sentiment_analysis_suite_5e67f5.py**

```
Error: unterminated string literal (detected at line 163) (<unknown>, line 163)
```

**test_astype_5e9cc9.py**

```
Error: '(' was never closed (<unknown>, line 290)
```

**test_base_indexer_9339e3.py**

```
Error: '[' was never closed (<unknown>, line 203)
```

**test_datetime_index_0892c0.py**

```
Error: '(' was never closed (<unknown>, line 248)
```

**test_find_replace_bc2ab9.py**

```
Error: unterminated string literal (detected at line 257) (<unknown>, line 257)
```

**test_http_parser_74f389.py**

```
Error: expected '(' (<unknown>, line 290)
```

**test_numba_246dfa.py**

```
Error: unterminated string literal (detected at line 227) (<unknown>, line 227)
```

**test_payments_dde789.py**

```
Error: '(' was never closed (<unknown>, line 210)
```

**test_sort_index_15410c.py**

```
Error: '{' was never closed (<unknown>, line 230)
```

**textual_entailment_suite_19763e.py**

```
Error: '(' was never closed (<unknown>, line 168)
```

**__init___14fc3d.py**

```
Error: invalid syntax (<unknown>, line 358)
```

**api_tests_b98962.py**

```
Error: expected '(' (<unknown>, line 275)
```

**beam_search_test_a8133f.py**

```
Error: '(' was never closed (<unknown>, line 185)
```

**cloud_storage_b743ea.py**

```
Error: '[' was never closed (<unknown>, line 448)
```

**common_ee0adc.py**

```
Error: unterminated triple-quoted string literal (detected at line 496) (<unknown>, line 491)
```

**dockerutils_6badca.py**

```
Error: '{' was never closed (<unknown>, line 397)
```

**fairness_metrics_fcc9f2.py**

```
Error: '[' was never closed (<unknown>, line 281)
```

**packager_d0d7be.py**

```
Error: unterminated string literal (detected at line 326) (<unknown>, line 326)
```

**realm_settings_e9d618.py**

```
Error: '(' was never closed (<unknown>, line 241)
```

**sas7bdat_f3d8c2.py**

```
Error: invalid syntax (<unknown>, line 354)
```

**schema_yaml_readers_9dabb0.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**server_057ee6.py**

```
Error: unterminated triple-quoted string literal (detected at line 268) (<unknown>, line 264)
```

**test_categorical_c93963.py**

```
Error: '(' was never closed (<unknown>, line 241)
```

**test_indexing_b72bb8.py**

```
Error: '(' was never closed (<unknown>, line 226)
```

**test_parse_dates_206fdc.py**

```
Error: '(' was never closed (<unknown>, line 189)
```

**test_raises_891508.py**

```
Error: '{' was never closed (<unknown>, line 163)
```

**test_setops_9f3297.py**

```
Error: unterminated string literal (detected at line 325) (<unknown>, line 325)
```

**test_usecols_basic_f18d3b.py**

```
Error: '(' was never closed (<unknown>, line 212)
```

**test_utils_014a22.py**

```
Error: invalid syntax (<unknown>, line 229)
```

**utils_1a8fd5.py**

```
Error: expected an indented block after 'if' statement on line 327 (<unknown>, line 327)
```

**test_channel_622ed8.py**

```
Error: invalid syntax (<unknown>, line 212)
```

**test_constructors_4c47aa.py**

```
Error: expected '(' (<unknown>, line 288)
```

**test_edge_d080a8.py**

```
Error: '{' was never closed (<unknown>, line 168)
```

**test_floats_b22095.py**

```
Error: '{' was never closed (<unknown>, line 283)
```

**test_flow_run_d3356c.py**

```
Error: unterminated string literal (detected at line 195) (<unknown>, line 195)
```

**test_index_d51bbf.py**

```
Error: '(' was never closed (<unknown>, line 228)
```

**test_indexing_235f4f.py**

```
Error: invalid syntax (<unknown>, line 242)
```

**test_merge_asof_031054.py**

```
Error: unterminated string literal (detected at line 63) (<unknown>, line 63)
```

**test_query_eval_130e95.py**

```
Error: '(' was never closed (<unknown>, line 298)
```

**test_reflection_3e7847.py**

```
Error: '(' was never closed (<unknown>, line 370)
```

**test_rolling_741e1f.py**

```
Error: '[' was never closed (<unknown>, line 190)
```

**test_to_csv_2a1c22.py**

```
Error: unterminated string literal (detected at line 240) (<unknown>, line 240)
```

**test_win_type_fcca97.py**

```
Error: '[' was never closed (<unknown>, line 185)
```

**transformation_82baef.py**

```
Error: '{' was never closed (<unknown>, line 383)
```

**web_rtc_831f2f.py**

```
Error: '(' was never closed (<unknown>, line 361)
```

**from_params_test_811442.py**

```
Error: unterminated string literal (detected at line 433) (<unknown>, line 433)
```

**indicators_960952.py**

```
Error: expected 'except' or 'finally' block (<unknown>, line 267)
```

**media_player_468e90.py**

```
Error: expected an indented block after function definition on line 358 (<unknown>, line 359)
```

**sorting_f49b49.py**

```
Error: unterminated triple-quoted string literal (detected at line 461) (<unknown>, line 457)
```

**string__8c20b1.py**

```
Error: '(' was never closed (<unknown>, line 382)
```

**test_aggregate_1eee57.py**

```
Error: '{' was never closed (<unknown>, line 280)
```

**test_cloud_storage_1d2f60.py**

```
Error: expected '(' (<unknown>, line 287)
```

**test_html_cda654.py**

```
Error: '(' was never closed (<unknown>, line 266)
```

**test_inference_f5d827.py**

```
Error: invalid syntax (<unknown>, line 193)
```

**test_rocksdb_e6756b.py**

```
Error: '{' was never closed (<unknown>, line 300)
```

**_compat_bfcdb3.py**

```
Error: expected ':' (<unknown>, line 327)
```

**auth_d8f17e.py**

```
Error: unterminated triple-quoted string literal (detected at line 356) (<unknown>, line 356)
```

**instance_410c65.py**

```
Error: unterminated triple-quoted string literal (detected at line 417) (<unknown>, line 414)
```

**object_array_68abb9.py**

```
Error: invalid syntax (<unknown>, line 401)
```

**run_b777ab.py**

```
Error: '(' was never closed (<unknown>, line 327)
```

**stdlib_4fd1ea.py**

```
Error: '(' was never closed (<unknown>, line 336)
```

**test_base_397a75.py**

```
Error: expected ':' (<unknown>, line 345)
```

**test_cli_3a2b0d.py**

```
Error: expected '(' (<unknown>, line 320)
```

**test_frame_color_e225c0.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**test_generic_7cde9d.py**

```
Error: '[' was never closed (<unknown>, line 311)
```

**test_kedro_data_catalog_4cfadd.py**

```
Error: unterminated string literal (detected at line 268) (<unknown>, line 268)
```

**test_process_sync_aggregate_c588af.py**

```
Error: '(' was never closed (<unknown>, line 240)
```

**test_resample_api_f0cdf1.py**

```
Error: unterminated string literal (detected at line 323) (<unknown>, line 323)
```

**test_rpc_0215ce.py**

```
Error: unterminated f-string literal (detected at line 186) (<unknown>, line 186)
```

**test_stateful_cf2044.py**

```
Error: expected '(' (<unknown>, line 514)
```

**test_wrappers_62b3f0.py**

```
Error: '(' was never closed (<unknown>, line 336)
```

**users_e2ac2d.py**

```
Error: unterminated triple-quoted string literal (detected at line 288) (<unknown>, line 287)
```

**conftest_445a11.py**

```
Error: unterminated string literal (detected at line 307) (<unknown>, line 307)
```

**kedro_data_catalog_148318.py**

```
Error: unterminated triple-quoted string literal (detected at line 386) (<unknown>, line 381)
```

**parquet_a441a6.py**

```
Error: unterminated triple-quoted string literal (detected at line 294) (<unknown>, line 256)
```

**test_cli_4e6ec5.py**

```
Error: expected '(' (<unknown>, line 242)
```

**test_datetime64_b94434.py**

```
Error: '[' was never closed (<unknown>, line 274)
```

**test_datetimelike_0b6f81.py**

```
Error: '(' was never closed (<unknown>, line 259)
```

**test_eth1_chaindb_433a8a.py**

```
Error: '(' was never closed (<unknown>, line 239)
```

**test_format_57a8f1.py**

```
Error: '(' was never closed (<unknown>, line 241)
```

**test_ipython_b46143.py**

```
Error: unterminated string literal (detected at line 296) (<unknown>, line 296)
```

**test_matrix_transport_fc1c06.py**

```
Error: unterminated triple-quoted string literal (detected at line 293) (<unknown>, line 291)
```

**test_pythonapi_d455c1.py**

```
Error: unterminated triple-quoted string literal (detected at line 233) (<unknown>, line 224)
```

**test_quantile_94afc3.py**

```
Error: '[' was never closed (<unknown>, line 225)
```

**test_replace_fc3342.py**

```
Error: unterminated string literal (detected at line 301) (<unknown>, line 301)
```

**test_rest_9e5773.py**

```
Error: unterminated string literal (detected at line 259) (<unknown>, line 259)
```

**test_stack_unstack_b54311.py**

```
Error: unterminated string literal (detected at line 210) (<unknown>, line 210)
```

**base_03e7f5.py**

```
Error: '[' was never closed (<unknown>, line 253)
```

**datastructures_0a7b11.py**

```
Error: '[' was never closed (<unknown>, line 441)
```

**header_990b83.py**

```
Error: '(' was never closed (<unknown>, line 326)
```

**params_226a6a.py**

```
Error: unterminated triple-quoted string literal (detected at line 419) (<unknown>, line 408)
```

**test_arithmetic_ef3dfe.py**

```
Error: expected ':' (<unknown>, line 298)
```

**test_base_worker_f71266.py**

```
Error: '[' was never closed (<unknown>, line 253)
```

**test_constructors_c6bb5c.py**

```
Error: '[' was never closed (<unknown>, line 299)
```

**test_consumer_4478ae.py**

```
Error: unterminated string literal (detected at line 338) (<unknown>, line 338)
```

**test_datetimes_630624.py**

```
Error: '[' was never closed (<unknown>, line 314)
```

**test_frame_60c296.py**

```
Error: unterminated string literal (detected at line 245) (<unknown>, line 245)
```

**test_iloc_211b3a.py**

```
Error: '[' was never closed (<unknown>, line 267)
```

**test_indexing_36b06e.py**

```
Error: '(' was never closed (<unknown>, line 232)
```

**test_numba_c3554f.py**

```
Error: unterminated string literal (detected at line 243) (<unknown>, line 243)
```

**test_ops_on_diff_frames_1c24fd.py**

```
Error: invalid syntax (<unknown>, line 228)
```

**test_replace_ac1a27.py**

```
Error: invalid syntax (<unknown>, line 246)
```

**test_selector_bbc47d.py**

```
Error: '(' was never closed (<unknown>, line 127)
```

**test_sort_values_87e12b.py**

```
Error: '(' was never closed (<unknown>, line 225)
```

**user_deposit_72a164.py**

```
Error: expected '(' (<unknown>, line 295)
```

**wrappers_34ca66.py**

```
Error: expected an indented block after function definition on line 433 (<unknown>, line 433)
```

**artifacts_099b0a.py**

```
Error: unterminated triple-quoted string literal (detected at line 452) (<unknown>, line 448)
```

**label_model_609423.py**

```
Error: unterminated triple-quoted string literal (detected at line 423) (<unknown>, line 394)
```

**switch_24834b.py**

```
Error: unterminated string literal (detected at line 304) (<unknown>, line 304)
```

**test_converter_f508e6.py**

```
Error: '[' was never closed (<unknown>, line 118)
```

**test_flags_92e1da.py**

```
Error: unterminated string literal (detected at line 292) (<unknown>, line 292)
```

**test_frame_plot_matplotlib_9669ca.py**

```
Error: unterminated string literal (detected at line 279) (<unknown>, line 279)
```

**test_melt_711512.py**

```
Error: '[' was never closed (<unknown>, line 420)
```

**test_modular_pipeline_544bcc.py**

```
Error: unterminated string literal (detected at line 216) (<unknown>, line 216)
```

**test_multi_209bc7.py**

```
Error: unterminated string literal (detected at line 184) (<unknown>, line 184)
```

**test_pivot_62e6c1.py**

```
Error: unterminated string literal (detected at line 212) (<unknown>, line 212)
```

**appgraph_e5732d.py**

```
Error: '(' was never closed (<unknown>, line 232)
```

**commands_tests_eb6597.py**

```
Error: '(' was never closed (<unknown>, line 233)
```

**datetimelike_f48bd1.py**

```
Error: '(' was never closed (<unknown>, line 431)
```

**fbeta_measure_test_d2fc57.py**

```
Error: expected an indented block after function definition on line 283 (<unknown>, line 283)
```

**options_fe74b9.py**

```
Error: '(' was never closed (<unknown>, line 441)
```

**s3_48c5ab.py**

```
Error: unterminated triple-quoted string literal (detected at line 483) (<unknown>, line 480)
```

**test_alt_backend_8ffa2d.py**

```
Error: '(' was never closed (<unknown>, line 391)
```

**test_blackouts_53e928.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**test_constraints_cdcb72.py**

```
Error: unterminated string literal (detected at line 269) (<unknown>, line 269)
```

**test_data_io_365c19.py**

```
Error: '(' was never closed (<unknown>, line 142)
```

**test_gen_data_e67a06.py**

```
Error: invalid syntax (<unknown>, line 278)
```

**test_microbatch_124d73.py**

```
Error: unterminated string literal (detected at line 172) (<unknown>, line 172)
```

**test_pairwise_9290e6.py**

```
Error: '(' was never closed (<unknown>, line 207)
```

**test_partial_86bf04.py**

```
Error: '[' was never closed (<unknown>, line 293)
```

**test_scalar_compat_34fd8d.py**

```
Error: '[' was never closed (<unknown>, line 258)
```

**test_shrinker_1f54e0.py**

```
Error: expected '(' (<unknown>, line 342)
```

**test_store_f376cd.py**

```
Error: unterminated string literal (detected at line 247) (<unknown>, line 247)
```

**test_validation_2c8d12.py**

```
Error: '{' was never closed (<unknown>, line 200)
```

**accessors_09b267.py**

```
Error: '(' was never closed (<unknown>, line 443)
```

**fan_63b94a.py**

```
Error: invalid syntax (<unknown>, line 232)
```

**forms_6e49d8.py**

```
Error: expected an indented block after 'if' statement on line 303 (<unknown>, line 303)
```

**initiator_manager_26765c.py**

```
Error: invalid syntax (<unknown>, line 236)
```

**jinja_context_test_42ca63.py**

```
Error: unterminated triple-quoted string literal (detected at line 314) (<unknown>, line 313)
```

**sqla_models_tests_7b5225.py**

```
Error: unterminated string literal (detected at line 198) (<unknown>, line 198)
```

**test_decimal_c85894.py**

```
Error: '(' was never closed (<unknown>, line 321)
```

**test_entity_registry_95c378.py**

```
Error: unterminated string literal (detected at line 183) (<unknown>, line 183)
```

**test_history_818171.py**

```
Error: unterminated string literal (detected at line 195) (<unknown>, line 195)
```

**test_init_75a7d5.py**

```
Error: expected an indented block after 'with' statement on line 249 (<unknown>, line 250)
```

**test_interval_3fb811.py**

```
Error: '[' was never closed (<unknown>, line 270)
```

**test_rank_c4ccb1.py**

```
Error: invalid syntax (<unknown>, line 218)
```

**test_reductions_08a693.py**

```
Error: invalid syntax (<unknown>, line 309)
```

**test_rolling_functions_251098.py**

```
Error: '(' was never closed (<unknown>, line 163)
```

**test_round_trip_b8b484.py**

```
Error: expected an indented block after function definition on line 262 (<unknown>, line 263)
```

**test_s3_fad4be.py**

```
Error: unterminated triple-quoted string literal (detected at line 307) (<unknown>, line 306)
```

**test_timedelta64_f628bf.py**

```
Error: unterminated string literal (detected at line 292) (<unknown>, line 292)
```

**test_to_latex_4e44c5.py**

```
Error: unterminated string literal (detected at line 223) (<unknown>, line 223)
```

**__init___3f6268.py**

```
Error: '{' was never closed (<unknown>, line 314)
```

**aiokafka_d62f1d.py**

```
Error: '[' was never closed (<unknown>, line 291)
```

**clienttrader_5e3b99.py**

```
Error: expected an indented block after 'except' statement on line 382 (<unknown>, line 383)
```

**coordinator_e4c72e.py**

```
Error: '(' was never closed (<unknown>, line 291)
```

**evaluators_b911b8.py**

```
Error: unterminated triple-quoted string literal (detected at line 422) (<unknown>, line 413)
```

**http_cookies_a8fd56.py**

```
Error: '(' was never closed (<unknown>, line 452)
```

**retention_81eb85.py**

```
Error: expected ':' (<unknown>, line 206)
```

**rewards_039ecf.py**

```
Error: expected '(' (<unknown>, line 292)
```

**splitters_9b6f18.py**

```
Error: '[' was never closed (<unknown>, line 314)
```

**task_engine_88162b.py**

```
Error: '{' was never closed (<unknown>, line 340)
```

**test_common_basic_a8eb4c.py**

```
Error: unterminated string literal (detected at line 183) (<unknown>, line 183)
```

**test_converter_orderflow_7dad6d.py**

```
Error: '(' was never closed (<unknown>, line 227)
```

**test_datetimelike_db9f17.py**

```
Error: '(' was never closed (<unknown>, line 317)
```

**test_libsparse_7cdef7.py**

```
Error: '(' was never closed (<unknown>, line 245)
```

**test_opcodes_be536f.py**

```
Error: unterminated string literal (detected at line 123) (<unknown>, line 123)
```

**test_rpc_telegram_deaea2.py**

```
Error: '(' was never closed (<unknown>, line 278)
```

**test_setitem_a50d5b.py**

```
Error: '[' was never closed (<unknown>, line 264)
```

**test_sparse_d759ae.py**

```
Error: invalid syntax (<unknown>, line 309)
```

**test_sql_f4958a.py**

```
Error: unterminated string literal (detected at line 285) (<unknown>, line 285)
```

**test_stoploss_on_exchange_98cce5.py**

```
Error: '(' was never closed (<unknown>, line 235)
```

**test_strings_4cd98f.py**

```
Error: '(' was never closed (<unknown>, line 230)
```

**util_test_ea6c40.py**

```
Error: '[' was never closed (<unknown>, line 166)
```

**email_mirror_590460.py**

```
Error: '[' was never closed (<unknown>, line 340)
```

**iterable_71c277.py**

```
Error: '(' was never closed (<unknown>, line 388)
```

**project_7870ca.py**

```
Error: unterminated triple-quoted string literal (detected at line 284) (<unknown>, line 284)
```

**sql_parse_tests_ebc947.py**

```
Error: unterminated triple-quoted string literal (detected at line 219) (<unknown>, line 218)
```

**stubs_b63ee1.py**

```
Error: '(' was never closed (<unknown>, line 381)
```

**test_arithmetic_f016de.py**

```
Error: invalid syntax (<unknown>, line 284)
```

**test_arrays_9fefe1.py**

```
Error: '[' was never closed (<unknown>, line 277)
```

**test_fillna_057fac.py**

```
Error: unterminated string literal (detected at line 248) (<unknown>, line 248)
```

**test_purge_979a37.py**

```
Error: expected ':' (<unknown>, line 280)
```

**test_readers_6fbe49.py**

```
Error: invalid syntax (<unknown>, line 279)
```

**test_sorting_c07df0.py**

```
Error: '[' was never closed (<unknown>, line 235)
```

**test_websocket_parser_3e2e83.py**

```
Error: '(' was never closed (<unknown>, line 285)
```

---

## Files with Changes (70 total)

### network_manager_8c67f5.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `Response`
- Class removed: `SecurityDetails`
- Function signature changed: `__init__`
  - Original params: `self, subjectName, issuer, validFrom, validTo, protocol`
  - Modified params: `self, client, requestId, interceptionId, isNavigationRequest, allowInterception, url, resourceType, payload, frame, redirectChain`
- Control flow structure modified:
  - `if_statements`: 42 -> 31 (-11 removed)
  - `for_loops`: 4 -> 2 (-2 removed)
  - `try_except`: 5 -> 2 (-3 removed)

### trainer_test_f0df2a.py

**Changes detected**: class_missing_in_typed, method_added_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `TestSparseClipGrad`
- Class removed: `SlowDataLoader`
- Class removed: `TestAmpTrainer`
- Class removed: `RecordMetricLearningRateScheduler`
- Class removed: `FakeOnBatchCallback`
- Class removed: `FakeTrainerCallback`
- Method added to `FakeModel`: `__init__`
- Function signature changed: `__init__`
  - Original params: `self, optimizer`
  - Modified params: `self, vocab`
- Control flow structure modified:
  - `if_statements`: 11 -> 2 (-9 removed)
  - `for_loops`: 24 -> 8 (-16 removed)
  - `with_statements`: 6 -> 2 (-4 removed)

### api_test_5baf5c.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `with_statements`: 5 -> 2 (-3 removed)

### datetimes_c4b343.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 8 -> 9 (+1 added)

### test_interface_94cb57.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 4 -> 0 (-4 removed)
  - `with_statements`: 17 -> 5 (-12 removed)

### augmented_lstm_c8e957.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `BiAugmentedLstm`: `_forward_unidirectional`
- Control flow structure modified:
  - `if_statements`: 17 -> 15 (-2 removed)
  - `for_loops`: 4 -> 3 (-1 removed)

### test_na_values_e137af.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 28 -> 21 (-7 removed)
  - `with_statements`: 17 -> 9 (-8 removed)

### expr_a7337f.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `PandasExprVisitor`
- Class removed: `Expr`
- Class removed: `PythonExprVisitor`
- Method removed from `BaseExprVisitor`: `visit_Assign`
- Method removed from `BaseExprVisitor`: `visit_Compare`
- Method removed from `BaseExprVisitor`: `visit_Index`
- Method removed from `BaseExprVisitor`: `visit_BoolOp`
- Method removed from `BaseExprVisitor`: `visit_Attribute`
- Method removed from `BaseExprVisitor`: `translate_In`
- Method removed from `BaseExprVisitor`: `visit_Call`
- Method removed from `BaseExprVisitor`: `_try_visit_binop`
- Method removed from `BaseExprVisitor`: `visit_Slice`
- Method removed from `BaseExprVisitor`: `visit_Subscript`
- Function signature changed: `__init__`
  - Original params: `self, expr, engine, parser, env, level`
  - Modified params: `self, env, engine, parser, preparser`
- Control flow structure modified:
  - `if_statements`: 42 -> 22 (-20 removed)
  - `for_loops`: 5 -> 3 (-2 removed)
  - `try_except`: 6 -> 1 (-5 removed)

### mediator_edd547.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 78 -> 30 (-48 removed)
  - `for_loops`: 17 -> 8 (-9 removed)

### users_e38fbd.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 32 -> 18 (-14 removed)
  - `for_loops`: 16 -> 12 (-4 removed)
  - `try_except`: 1 -> 0 (-1 removed)

### value_05eb19.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 39 -> 35 (-4 removed)
  - `for_loops`: 10 -> 7 (-3 removed)
  - `while_loops`: 1 -> 0 (-1 removed)

### classes_29984e.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `BaseSignature`
- Class removed: `Name`
- Class removed: `Completion`
- Class removed: `ParamName`
- Class removed: `Signature`
- Method removed from `BaseName`: `execute`
- Method removed from `BaseName`: `get_type_hint`
- Method removed from `BaseName`: `get_line_code`
- Method removed from `BaseName`: `__repr__`
- Method removed from `BaseName`: `get_signatures`
- Method removed from `BaseName`: `_get_signatures`
- Function signature changed: `defined_names`
  - Original params: `self`
  - Modified params: `inference_state, value`
- Control flow structure modified:
  - `if_statements`: 43 -> 29 (-14 removed)
  - `while_loops`: 1 -> 0 (-1 removed)

### deployer_c2beb1.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `NoopResultsRecorder`
- Class removed: `BuildStage`
- Class removed: `PolicyGenerator`
- Class removed: `DeploymentReporter`
- Class removed: `ResultsRecorder`
- Class removed: `WebsocketPolicyInjector`
- Method removed from `LambdaEventSourcePolicyInjector`: `_needs_policy_injected`
- Method removed from `LambdaEventSourcePolicyInjector`: `handle_dynamodbeventsource`
- Method removed from `LambdaEventSourcePolicyInjector`: `_inject_trigger_policy`
- Method removed from `LambdaEventSourcePolicyInjector`: `handle_kinesiseventsource`
- Method removed from `LambdaEventSourcePolicyInjector`: `handle_sqseventsource`
- Function signature changed: `__init__`
  - Original params: `self, ui`
  - Modified params: `self`
- Control flow structure modified:
  - `if_statements`: 27 -> 17 (-10 removed)
  - `for_loops`: 3 -> 0 (-3 removed)
  - `try_except`: 4 -> 3 (-1 removed)

### import_export_tests_dc500f.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestImportExport`: `test_import_table_override`
- Method removed from `TestImportExport`: `test_import_table_1_col_1_met`
- Method removed from `TestImportExport`: `test_import_table_no_metadata`
- Method removed from `TestImportExport`: `_create_dashboard_for_import`
- Method removed from `TestImportExport`: `test_import_table_2_col_2_met`
- Method removed from `TestImportExport`: `test_import_table_override_identical`
- Method removed from `TestImportExport`: `test_import_new_dashboard_slice_reset_ownership`
- Method removed from `TestImportExport`: `test_import_override_dashboard_2_slices`
- Method removed from `TestImportExport`: `test_import_override_dashboard_slice_reset_ownership`
- Control flow structure modified:
  - `if_statements`: 10 -> 9 (-1 removed)

### join_merge_5c5786.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed

#### What Changed:

- Class removed: `MergeMultiIndex`
- Class removed: `MergeOrdered`
- Class removed: `Align`
- Class removed: `MergeAsof`
- Method removed from `MergeCategoricals`: `time_merge_cat`
- Method removed from `MergeCategoricals`: `time_merge_on_cat_idx`
- Method removed from `MergeCategoricals`: `time_merge_on_cat_col`

### messages_bf1e5c.py

**Changes detected**: method_added_in_typed, parameter_mismatch

#### What Changed:

- Method added to `ReturnWithArgsInsideGenerator`: `__init__`
- Function signature changed: `__init__`
  - Original params: `self, filename, loc, missing_arguments`
  - Modified params: `self, filename, loc`

### multi_7dfe48.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `MultiIndex`: `hasnans`
- Method removed from `MultiIndex`: `factorize`
- Method removed from `MultiIndex`: `intersection`
- Method removed from `MultiIndex`: `asi8`
- Method removed from `MultiIndex`: `__iter__`
- Method removed from `MultiIndex`: `item`
- Method removed from `MultiIndex`: `inferred_type`
- Method removed from `MultiIndex`: `get_level_values`
- Method removed from `MultiIndex`: `insert`
- Control flow structure modified:
  - `if_statements`: 41 -> 32 (-9 removed)

### test_arithmetics_2d6e2b.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestSparseArrayArithmetics`: `test_mixed_array_float_int`
- Method removed from `TestSparseArrayArithmetics`: `test_bool_same_index`
- Method removed from `TestSparseArrayArithmetics`: `test_mixed_array_comparison`
- Method removed from `TestSparseArrayArithmetics`: `test_bool_array_logical`
- Method removed from `TestSparseArrayArithmetics`: `test_xor`
- Control flow structure modified:
  - `try_except`: 1 -> 0 (-1 removed)
  - `with_statements`: 5 -> 2 (-3 removed)

### __init___7ab01a.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `PyRegExp`
- Method removed from `Regex`: `finditer`
- Method removed from `Regex`: `subn`
- Method removed from `Regex`: `sub`
- Function signature changed: `__init__`
  - Original params: `self, pyPattern, flags`
  - Modified params: `self, pattern, flags`
- Control flow structure modified:
  - `if_statements`: 60 -> 49 (-11 removed)
  - `while_loops`: 2 -> 1 (-1 removed)

### test_backtesting_c617d3.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 35 -> 5 (-30 removed)
  - `for_loops`: 18 -> 4 (-14 removed)
  - `with_statements`: 9 -> 3 (-6 removed)

### test_base_fc0d64.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `test_Collection`: `test_to_value`
- Method removed from `test_Collection`: `test_label`
- Method removed from `test_Collection`: `test_relative_now`
- Method removed from `test_Collection`: `test_windowed_now`
- Method removed from `test_Collection`: `test_relative_field`
- Method removed from `test_Collection`: `test_and`
- Method removed from `test_Collection`: `test_repr_info`
- Method removed from `test_Collection`: `test_windowed_delta`
- Method removed from `test_Collection`: `test_relative_timestamp`
- Method removed from `test_Collection`: `test__maybe_del_key_ttl`
- Method removed from `test_Collection`: `test__maybe_set_key_ttl`
- Method removed from `test_Collection`: `test__human_channel`
- Method removed from `test_Collection`: `test_shortlabel`
- Method removed from `test_Collection`: `test_windowed_timestamp`
- Method removed from `test_Collection`: `test_to_key`
- Method removed from `test_Collection`: `test_relative_event`
- Method removed from `test_Collection`: `test_apply_changelog_batch`
- Method removed from `test_Collection`: `mock_ranges`
- Method removed from `test_Collection`: `test_partition_for_key__partitioner`
- Method removed from `test_Collection`: `test_relative_field__raises_if_no_event`
- Method removed from `test_Collection`: `test_window_ranges`
- Method removed from `test_Collection`: `test_relative_now__no_event`
- Method removed from `test_Collection`: `test_set_del_windowed`
- Method removed from `test_Collection`: `test_relative_event__raises_if_no_event`
- Method removed from `test_Collection`: `test_apply_window_op`
- Control flow structure modified:
  - `for_loops`: 2 -> 0 (-2 removed)
  - `with_statements`: 11 -> 4 (-7 removed)

### test_cut_a5ac0a.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 6 -> 2 (-4 removed)
  - `with_statements`: 12 -> 7 (-5 removed)

### test_process_pending_consolidations_a9c6ec.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 7 -> 0 (-7 removed)
  - `for_loops`: 9 -> 3 (-6 removed)

### test_series_apply_554287.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 9 -> 8 (-1 removed)
  - `with_statements`: 15 -> 13 (-2 removed)

### context_54b9a9.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `quoted_str`
- Function signature changed: `wrapper`
  - Original params: ``
  - Modified params: `spec`
- Control flow structure modified:
  - `if_statements`: 34 -> 13 (-21 removed)
  - `for_loops`: 8 -> 2 (-6 removed)
  - `try_except`: 3 -> 2 (-1 removed)

### test_c_parser_only_bf97b6.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `NoNextBuffer`
- Control flow structure modified:
  - `with_statements`: 18 -> 7 (-11 removed)

### test_deps_61aa2a.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `MockRegistry`
- Class removed: `TestPackageSpec`
- Method removed from `TestHubPackage`: `test_resolve_ranges_install_prerelease_true`
- Method removed from `TestHubPackage`: `test_resolve_ranges_install_prerelease_default_false`
- Method removed from `TestHubPackage`: `test_get_version_latest_prelease_false`
- Method removed from `TestHubPackage`: `test_resolve_ranges`
- Method removed from `TestHubPackage`: `test_get_version_prerelease_explicitly_requested`
- Method removed from `TestHubPackage`: `test_get_version_latest_prelease_true`
- Control flow structure modified:
  - `try_except`: 2 -> 0 (-2 removed)
  - `with_statements`: 15 -> 11 (-4 removed)

### test_expanding_5d3564.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 9 -> 6 (-3 removed)
  - `with_statements`: 4 -> 2 (-2 removed)

### test_helpers_38220e.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 24 -> 22 (-2 removed)
  - `for_loops`: 8 -> 6 (-2 removed)
  - `with_statements`: 12 -> 6 (-6 removed)

### test_setitem_31620d.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestCoercionObject`
- Class removed: `TestPeriodIntervalCoercion`
- Class removed: `TestSetitemCallable`
- Class removed: `TestCoercionDatetime64`
- Class removed: `TestCoercionFloat32`
- Class removed: `TestCoercionFloat64`
- Class removed: `TestSetitemRangeIntoIntegerSeries`
- Class removed: `TestSetitemDT64IntoInt`
- Class removed: `TestSeriesNoneCoercion`
- Class removed: `TestSmallIntegerSetitemUpcast`
- Class removed: `TestCoercionDatetime64TZ`
- Class removed: `TestCoercionDatetime64HigherReso`
- Class removed: `TestSetitemMismatchedTZCastsToObject`
- Class removed: `TestSetitemIntoIntegerSeriesNeedsUpcast`
- Class removed: `TestSetitemWithExpansion`
- Class removed: `TestSetitemCastingEquivalents`
- Class removed: `TestSetitemNAPeriodDtype`
- Class removed: `TestCoercionInt8`
- Class removed: `TestCoercionComplex`
- Class removed: `SetitemCastingEquivalents`
- Class removed: `TestCoercionString`
- Class removed: `TestSetitemNADatetimeLikeDtype`
- Class removed: `TestSetitemFloatNDarrayIntoIntegerSeries`
- Class removed: `TestSetitemFloatIntervalWithIntIntervalValues`
- Class removed: `TestCoercionInt64`
- Class removed: `TestCoercionBool`
- Class removed: `TestSetitemCasting`
- Class removed: `CoercionTest`
- Class removed: `TestSetitemTimedelta64IntoNumeric`
- Class removed: `TestCoercionTimedelta64`
- Control flow structure modified:
  - `if_statements`: 30 -> 0 (-30 removed)
  - `for_loops`: 2 -> 0 (-2 removed)
  - `with_statements`: 51 -> 9 (-42 removed)

### test_core_9112bf.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `TestLegacyLoadAndSave`
- Class removed: `MyLegacyVersionedDataset`
- Class removed: `MyLegacyDataset`
- Method removed from `TestAbstractVersionedDataset`: `test_cache_release`
- Function signature changed: `test_save_version_warning`
  - Original params: `self, my_legacy_versioned_dataset, load_version, save_version, dummy_data`
  - Modified params: `self, my_versioned_dataset, load_version, save_version, dummy_data`
- Control flow structure modified:
  - `try_except`: 3 -> 2 (-1 removed)
  - `with_statements`: 22 -> 15 (-7 removed)

### test_alerts_3e2087.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `DummyRemoteIPPlugin`
- Method removed from `AlertsTestCase`: `test_alert_tagging`
- Method removed from `AlertsTestCase`: `test_duplicate_status`
- Method removed from `AlertsTestCase`: `test_history_limit`
- Method removed from `AlertsTestCase`: `test_alert_no_attributes`
- Method removed from `AlertsTestCase`: `test_alerts_show_fields`
- Method removed from `AlertsTestCase`: `test_alert_attributes`
- Method removed from `AlertsTestCase`: `test_duplicate_value`
- Method removed from `AlertsTestCase`: `test_get_body`
- Method removed from `AlertsTestCase`: `test_timeout`
- Method removed from `AlertsTestCase`: `test_filter_and_query_params`
- Method removed from `AlertsTestCase`: `test_filter_params`
- Method removed from `AlertsTestCase`: `test_query_param`
- Control flow structure modified:
  - `with_statements`: 1 -> 0 (-1 removed)

### test_integration_ae2f32.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 5 -> 0 (-5 removed)

### base_value_caa0cc.py

**Changes detected**: method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Method removed from `ValueSet`: `gather_annotation_classes`
- Method removed from `ValueSet`: `try_merge`
- Method removed from `ValueSet`: `get_type_hint`
- Method removed from `ValueSet`: `goto`
- Method removed from `ValueSet`: `get_signatures`
- Method removed from `ValueSet`: `get_item`
- Method removed from `ValueSet`: `py__getattribute__`
- Method removed from `ValueSet`: `infer_type_vars`
- Function signature changed: `goto`
  - Original params: `self`
  - Modified params: `self, name_or_str, name_context, analysis_errors`
- Control flow structure modified:
  - `if_statements`: 22 -> 17 (-5 removed)
  - `for_loops`: 8 -> 6 (-2 removed)
  - `try_except`: 3 -> 2 (-1 removed)

### bias_metrics_5f20ad.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `AssociationWithoutGroundTruth`
- Method removed from `NaturalLanguageInference`: `reset`
- Function signature changed: `__call__`
  - Original params: `self, predicted_labels, protected_variable_labels, mask`
  - Modified params: `self, nli_probabilities`
- Control flow structure modified:
  - `if_statements`: 26 -> 11 (-15 removed)
  - `for_loops`: 4 -> 1 (-3 removed)

### test_init_f7cb07.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `MockLightEntityEntity`
- Control flow structure modified:
  - `if_statements`: 6 -> 1 (-5 removed)
  - `for_loops`: 4 -> 1 (-3 removed)
  - `with_statements`: 14 -> 4 (-10 removed)

### test_series_5f79c1.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestSeriesPlots`: `test_invalid_kind`
- Method removed from `TestSeriesPlots`: `test_errorbar_plot_invalid_yerr_shape`
- Method removed from `TestSeriesPlots`: `test_plot_fails_with_dupe_color_and_style`
- Method removed from `TestSeriesPlots`: `test_df_series_secondary_legend`
- Method removed from `TestSeriesPlots`: `test_pie_arrow_type`
- Method removed from `TestSeriesPlots`: `test_pie_series_labels_and_colors`
- Method removed from `TestSeriesPlots`: `test_table_true`
- Method removed from `TestSeriesPlots`: `test_kind_attr`
- Method removed from `TestSeriesPlots`: `test_secondary_logy`
- Method removed from `TestSeriesPlots`: `test_series_grid_settings`
- Method removed from `TestSeriesPlots`: `test_table_self`
- Method removed from `TestSeriesPlots`: `test_time_series_plot_color_with_empty_kwargs`
- Method removed from `TestSeriesPlots`: `test_xtick_barPlot`
- Method removed from `TestSeriesPlots`: `test_df_series_secondary_legend_both`
- Method removed from `TestSeriesPlots`: `test_invalid_plot_data`
- Method removed from `TestSeriesPlots`: `test_kde_missing_vals`
- Method removed from `TestSeriesPlots`: `test_secondary_y_subplot_axis_labels`
- Method removed from `TestSeriesPlots`: `test_kind_kwarg`
- Method removed from `TestSeriesPlots`: `test_errorbar_asymmetrical`
- Method removed from `TestSeriesPlots`: `test_errorbar_plot_invalid_yerr`
- Method removed from `TestSeriesPlots`: `test_density_kwargs`
- Method removed from `TestSeriesPlots`: `test_style_single_ok`
- Method removed from `TestSeriesPlots`: `test_errorbar_asymmetrical_error`
- Method removed from `TestSeriesPlots`: `test_pie_series_negative_raises`
- Method removed from `TestSeriesPlots`: `test_kde_kwargs_weights`
- Method removed from `TestSeriesPlots`: `test_pie_series_nan`
- Method removed from `TestSeriesPlots`: `test_plot_xlim_for_series`
- Method removed from `TestSeriesPlots`: `test_df_series_secondary_legend_both_with_axis_2`
- Method removed from `TestSeriesPlots`: `test_partially_invalid_plot_data`
- Method removed from `TestSeriesPlots`: `test_plot_no_numeric_data`
- Method removed from `TestSeriesPlots`: `test_plot_order`
- Method removed from `TestSeriesPlots`: `test_plot_no_rows`
- Method removed from `TestSeriesPlots`: `test_pie_nan`
- Method removed from `TestSeriesPlots`: `test_xticklabels`
- Method removed from `TestSeriesPlots`: `test_custom_business_day_freq`
- Method removed from `TestSeriesPlots`: `test_xlabel_ylabel_series`
- Method removed from `TestSeriesPlots`: `test_errorbar_plot_ts`
- Method removed from `TestSeriesPlots`: `test_standard_colors`
- Method removed from `TestSeriesPlots`: `test_plot_accessor_updates_on_inplace`
- Method removed from `TestSeriesPlots`: `test_boxplot_series`
- Method removed from `TestSeriesPlots`: `test_series_none_color`
- Method removed from `TestSeriesPlots`: `test_pie_series_no_label`
- Method removed from `TestSeriesPlots`: `test_errorbar_plot_yerr_0`
- Method removed from `TestSeriesPlots`: `test_kde_kwargs`
- Method removed from `TestSeriesPlots`: `test_standard_colors_all`
- Method removed from `TestSeriesPlots`: `test_valid_object_plot`
- Method removed from `TestSeriesPlots`: `test_time_series_plot_color_kwargs`
- Method removed from `TestSeriesPlots`: `test_pie_series_autopct_and_fontsize`
- Method removed from `TestSeriesPlots`: `test_errorbar_plot`
- Method removed from `TestSeriesPlots`: `test_pie_series_less_colors_than_elements`
- Method removed from `TestSeriesPlots`: `test_series_plot_color_kwargs`
- Method removed from `TestSeriesPlots`: `test_plot_no_warning`
- Method removed from `TestSeriesPlots`: `test_kde_kwargs_check_axes`
- Method removed from `TestSeriesPlots`: `test_dup_datetime_index_plot`
- Method removed from `TestSeriesPlots`: `test_timedelta_index`
- Control flow structure modified:
  - `if_statements`: 2 -> 0 (-2 removed)
  - `for_loops`: 4 -> 0 (-4 removed)
  - `with_statements`: 11 -> 1 (-10 removed)

### test_session_extension_hooks_7b3b93.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestAsyncNodeDatasetHooks`
- Class removed: `TestKedroContextSpecsHook`
- Class removed: `LogCatalog`
- Method removed from `TestBeforeNodeRunHookWithInputUpdates`: `test_broken_input_update`
- Method removed from `TestBeforeNodeRunHookWithInputUpdates`: `test_correct_input_update_parallel`
- Method removed from `TestBeforeNodeRunHookWithInputUpdates`: `test_broken_input_update_parallel`
- Control flow structure modified:
  - `if_statements`: 1 -> 0 (-1 removed)
  - `with_statements`: 5 -> 3 (-2 removed)

### test_token_network_03ccab.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `for_loops`: 3 -> 1 (-2 removed)
  - `with_statements`: 26 -> 18 (-8 removed)

### ewmh_1d4730.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `EWMH`: `getWritableProperties`
- Method removed from `EWMH`: `getReadableProperties`
- Method removed from `EWMH`: `getProperty`
- Method removed from `EWMH`: `_setProperty`
- Method removed from `EWMH`: `setProperty`
- Method removed from `EWMH`: `_createWindow`
- Method removed from `EWMH`: `_getAtomName`
- Control flow structure modified:
  - `if_statements`: 18 -> 12 (-6 removed)
  - `try_except`: 1 -> 0 (-1 removed)

### fbeta_multi_label_measure_test_75eeda.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `FBetaMultiLabelMeasureTest`: `test_multiple_distributed_runs`
- Method removed from `FBetaMultiLabelMeasureTest`: `test_distributed_fbeta_multilabel_measure`
- Control flow structure modified:
  - `for_loops`: 3 -> 0 (-3 removed)

### objects_2ea8dd.py

**Changes detected**: parameter_mismatch

#### What Changed:

- Function signature changed: `get_window_bounds`
  - Original params: `self, num_values, min_periods, center, closed, step`
  - Modified params: `self, num_values, min_periods, center, closed, step, win_type`

### test_range_fc6fe9.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestRangeIndex`: `test_engineless_lookup`
- Method removed from `TestRangeIndex`: `test_isin_range`
- Method removed from `TestRangeIndex`: `test_sort_values_key`
- Method removed from `TestRangeIndex`: `test_append_len_one`
- Method removed from `TestRangeIndex`: `test_append`
- Method removed from `TestRangeIndex`: `test_range_index_rsub_by_const`
- Control flow structure modified:
  - `if_statements`: 3 -> 1 (-2 removed)
  - `with_statements`: 9 -> 5 (-4 removed)

### test_settlement_0a3fe7.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `with_statements`: 19 -> 2 (-17 removed)

### __init___4ea406.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `PanelRespons`
- Class removed: `IndexView`
- Class removed: `ManifestJSONView`
- Function signature changed: `__init__`
  - Original params: `self, repo_path, hass`
  - Modified params: `self, component_name, sidebar_title, sidebar_icon, frontend_url_path, config, require_admin, config_panel_domain`
- Control flow structure modified:
  - `if_statements`: 23 -> 12 (-11 removed)
  - `for_loops`: 4 -> 3 (-1 removed)
  - `with_statements`: 1 -> 0 (-1 removed)

### test_abstract_0be2c2.py

**Changes detected**: parameter_mismatch

#### What Changed:

- Function signature changed: `__exit__`
  - Original params: `self`
  - Modified params: `self, exc_type, exc_val, exc_tb`

### test_api_0e252b.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `IntegrationTests`: `test_goodFile`
- Method removed from `IntegrationTests`: `tearDown`
- Method removed from `IntegrationTests`: `test_errors`
- Method removed from `IntegrationTests`: `test_fileWithFlakes`
- Method removed from `IntegrationTests`: `getPyflakesBinary`
- Method removed from `IntegrationTests`: `test_readFromStdin`
- Method removed from `IntegrationTests`: `runPyflakes`
- Control flow structure modified:
  - `if_statements`: 6 -> 4 (-2 removed)

### test_base_3a84cb.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestIndexUtils`
- Class removed: `TestMixedIntIndex`
- Method removed from `TestIndex`: `test_equals_op_index_vs_mi_same_length`
- Method removed from `TestIndex`: `test_drop_by_str_label_errors_ignore`
- Method removed from `TestIndex`: `test_index_diff`
- Method removed from `TestIndex`: `test_map_dictlike`
- Method removed from `TestIndex`: `test_reindex_doesnt_preserve_type_if_target_is_empty_index`
- Method removed from `TestIndex`: `test_reindex_preserves_type_if_target_is_empty_list_or_array`
- Method removed from `TestIndex`: `test_is_numeric`
- Method removed from `TestIndex`: `test_summary`
- Method removed from `TestIndex`: `test_groupby`
- Method removed from `TestIndex`: `test_dt_conversion_preserves_name`
- Method removed from `TestIndex`: `test_isin_empty`
- Method removed from `TestIndex`: `test_tab_complete_warning`
- Method removed from `TestIndex`: `test_sortlevel_na_position`
- Method removed from `TestIndex`: `test_is_monotonic_incomparable`
- Method removed from `TestIndex`: `test_cached_properties_not_settable`
- Method removed from `TestIndex`: `test_outer_join_sort`
- Method removed from `TestIndex`: `test_map_tseries_indices_accsr_return_index`
- Method removed from `TestIndex`: `test_reindex_no_type_preserve_target_empty_mi`
- Method removed from `TestIndex`: `test_boolean_cmp`
- Method removed from `TestIndex`: `test_slice_is_unique`
- Method removed from `TestIndex`: `test_drop_with_duplicates_in_index`
- Method removed from `TestIndex`: `test_isin_nan_common_float64`
- Method removed from `TestIndex`: `test_take_bad_bounds_raises`
- Method removed from `TestIndex`: `test_slice_is_montonic`
- Method removed from `TestIndex`: `test_get_level_values`
- Method removed from `TestIndex`: `test_contains_method_removed`
- Method removed from `TestIndex`: `test_tab_completion`
- Method removed from `TestIndex`: `test_join_self`
- Method removed from `TestIndex`: `test_drop_tuple`
- Method removed from `TestIndex`: `test_str_bool_return`
- Method removed from `TestIndex`: `test_equals_op_multiindex_identify`
- Method removed from `TestIndex`: `test_take_fill_value_none_raises`
- Method removed from `TestIndex`: `test_map_tseries_indices_return_index`
- Method removed from `TestIndex`: `test_drop_by_numeric_label_loc`
- Method removed from `TestIndex`: `test_equals_op_mismatched_multiindex_raises`
- Method removed from `TestIndex`: `test_str_attribute_raises`
- Method removed from `TestIndex`: `test_sortlevel`
- Method removed from `TestIndex`: `test_isin_nan_common_object`
- Method removed from `TestIndex`: `test_map_na_exclusion`
- Method removed from `TestIndex`: `test_isin`
- Method removed from `TestIndex`: `test_drop_by_numeric_label_raises_missing_keys`
- Method removed from `TestIndex`: `test_index_round`
- Method removed from `TestIndex`: `test_drop_by_str_label_raises_missing_keys`
- Method removed from `TestIndex`: `test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric`
- Method removed from `TestIndex`: `test_is_object`
- Method removed from `TestIndex`: `test_slice_keep_name`
- Method removed from `TestIndex`: `test_indexing_doesnt_change_class`
- Method removed from `TestIndex`: `test_map_with_tuples_mi`
- Method removed from `TestIndex`: `test_str_attribute`
- Method removed from `TestIndex`: `test_drop_by_str_label`
- Method removed from `TestIndex`: `test_isin_level_kwarg_bad_level_raises`
- Method removed from `TestIndex`: `test_take_fill_value`
- Method removed from `TestIndex`: `test_map_dictlike_simple`
- Method removed from `TestIndex`: `test_isin_string_null`
- Method removed from `TestIndex`: `test_reindex_preserves_name_if_target_is_list_or_ndarray`
- Method removed from `TestIndex`: `test_isin_level_kwarg_bad_label_raises`
- Method removed from `TestIndex`: `test_isin_level_kwarg`
- Method removed from `TestIndex`: `test_reindex_ignoring_level`
- Method removed from `TestIndex`: `test_str_split`
- Method removed from `TestIndex`: `test_str_bool_series_indexing`
- Method removed from `TestIndex`: `test_map_defaultdict`
- Method removed from `TestIndex`: `test_equals_op_multiindex`
- Method removed from `TestIndex`: `test_map_with_tuples`
- Method removed from `TestIndex`: `test_logical_compat`
- Method removed from `TestIndex`: `test_append_empty_preserve_name`
- Method removed from `TestIndex`: `test_drop_by_numeric_label_errors_ignore`
- Method removed from `TestIndex`: `test_map_with_non_function_missing_values`
- Control flow structure modified:
  - `if_statements`: 34 -> 16 (-18 removed)
  - `for_loops`: 6 -> 3 (-3 removed)
  - `with_statements`: 45 -> 12 (-33 removed)

### test_common_27891c.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestError`
- Method removed from `TestMMapWrapper`: `test_binary_mode`
- Method removed from `TestMMapWrapper`: `test_unknown_engine`
- Method removed from `TestMMapWrapper`: `test_warning_missing_utf_bom`
- Control flow structure modified:
  - `if_statements`: 9 -> 6 (-3 removed)
  - `with_statements`: 50 -> 21 (-29 removed)

### test_frame_apply_391b73.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 24 -> 8 (-16 removed)
  - `for_loops`: 3 -> 0 (-3 removed)
  - `with_statements`: 15 -> 5 (-10 removed)

### test_plugins_ab0c47.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `Model2`
- Class removed: `Model1`
- Function signature changed: `foo`
  - Original params: ``
  - Modified params: `a`
- Control flow structure modified:
  - `with_statements`: 18 -> 16 (-2 removed)

### btanalysis_7decb4.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 45 -> 39 (-6 removed)

### notify_014492.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 23 -> 22 (-1 removed)

### test_arithmetic_e20900.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `TestFrameArithmeticUnsorted`
- Class removed: `SubclassedSeries`
- Class removed: `TestFrameArithmetic`
- Class removed: `TestFrameFlexArithmetic`
- Class removed: `SubclassedDataFrame`
- Function signature changed: `__init__`
  - Original params: `self, my_extra_data`
  - Modified params: `self, value, dtype`
- Control flow structure modified:
  - `if_statements`: 21 -> 2 (-19 removed)
  - `for_loops`: 7 -> 0 (-7 removed)
  - `with_statements`: 53 -> 17 (-36 removed)

### test_json_7e4832.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestJSONArray`: `test_EA_types`
- Method removed from `TestJSONArray`: `test_setitem_2d_values`
- Control flow structure modified:
  - `if_statements`: 8 -> 7 (-1 removed)
  - `for_loops`: 1 -> 0 (-1 removed)
  - `with_statements`: 4 -> 2 (-2 removed)

### test_offsets_595cf3.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestOffsetAliases`
- Class removed: `TestDateOffset`
- Class removed: `TestReprNames`
- Class removed: `TestOffsetNames`
- Method removed from `TestCommon`: `test_offsets_hashable`
- Method removed from `TestCommon`: `test_pickle_dateoffset_odd_inputs`
- Method removed from `TestCommon`: `test_add`
- Method removed from `TestCommon`: `test_add_empty_datetimeindex`
- Method removed from `TestCommon`: `test_pickle_roundtrip`
- Method removed from `TestCommon`: `test_add_dt64_ndarray_non_nano`
- Control flow structure modified:
  - `if_statements`: 21 -> 13 (-8 removed)
  - `for_loops`: 20 -> 8 (-12 removed)
  - `with_statements`: 25 -> 5 (-20 removed)

### test_transform_0bcea9.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 56 -> 6 (-50 removed)
  - `for_loops`: 12 -> 4 (-8 removed)
  - `with_statements`: 17 -> 7 (-10 removed)

### test_wallets_fe5a94.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 3 -> 2 (-1 removed)
  - `for_loops`: 1 -> 0 (-1 removed)

### cover_ccbdb1.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `PowerViewShadeTDBUBottom`
- Class removed: `PowerViewShadeDualOverlappedCombinedTilt`
- Class removed: `PowerViewShadeTDBUTop`
- Class removed: `PowerViewShadeDualOverlappedRear`
- Class removed: `PowerViewShadeDualRailBase`
- Class removed: `PowerViewShadeDualOverlappedFront`
- Class removed: `PowerViewShadeTopDown`
- Class removed: `PowerViewShadeDualOverlappedBase`
- Class removed: `PowerViewShadeDualOverlappedCombined`
- Method removed from `PowerViewShadeTiltOnly`: `is_closed`
- Method removed from `PowerViewShadeTiltOnly`: `transition_steps`
- Method removed from `PowerViewShadeTiltOnly`: `current_cover_position`
- Control flow structure modified:
  - `if_statements`: 10 -> 7 (-3 removed)

### util_e9973a.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 42 -> 35 (-7 removed)
  - `for_loops`: 4 -> 2 (-2 removed)
  - `try_except`: 5 -> 2 (-3 removed)

### __init___db92f1.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 17 -> 12 (-5 removed)
  - `try_except`: 3 -> 1 (-2 removed)
  - `with_statements`: 3 -> 1 (-2 removed)

### test_logging_b7910e.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `ServerWorkerTestImpl`
- Class removed: `TestWorkerLogging`
- Class removed: `TestPrefectConsoleHandler`
- Class removed: `TestAPILogWorker`
- Class removed: `CloudWorkerTestImpl`
- Class removed: `TestJsonFormatter`
- Class removed: `TestObfuscateApiKeyFilter`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_outside_of_run_context_with_default_setting`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_that_opt_out`
- Method removed from `TestAPILogHandler`: `test_does_not_raise_when_logger_outside_of_run_context_with_default_setting`
- Method removed from `TestAPILogHandler`: `test_does_not_warn_when_logger_outside_of_run_context_with_error_setting`
- Method removed from `TestAPILogHandler`: `test_does_not_raise_when_logger_outside_of_run_context_with_warn_setting`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_outside_of_run_context_with_warn_setting`
- Method removed from `TestAPILogHandler`: `test_missing_context_warning_refers_to_caller_lineno`
- Method removed from `TestAPILogHandler`: `test_does_not_raise_or_warn_when_logger_outside_of_run_context_with_ignore_setting`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_outside_of_run_context_with_ignore_setting`
- Method removed from `TestAPILogHandler`: `test_handler_knows_how_large_logs_are`
- Method removed from `TestAPILogHandler`: `test_does_not_write_error_for_logs_outside_run_context_that_opt_out`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_when_handler_is_disabled`
- Method removed from `TestAPILogHandler`: `test_does_not_send_logs_outside_of_run_context_with_error_setting`
- Method removed from `TestAPILogHandler`: `test_writes_logging_errors_to_stderr`
- Function signature changed: `my_flow`
  - Original params: ``
  - Modified params: `loggers`
- Control flow structure modified:
  - `if_statements`: 7 -> 2 (-5 removed)
  - `for_loops`: 9 -> 2 (-7 removed)
  - `try_except`: 2 -> 0 (-2 removed)
  - `with_statements`: 67 -> 14 (-53 removed)

### test_matrix_transport_afcccc.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `try_except`: 1 -> 0 (-1 removed)
  - `with_statements`: 2 -> 1 (-1 removed)

### test_reindex_54a65d.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestDataFrameSelectReindex`: `test_reindex_multi`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_empty_frame`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_single_column_ea_index_and_columns`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_not_category`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_corner`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_columns_method`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_single_named_indexer`
- Method removed from `TestDataFrameSelectReindex`: `test_non_monotonic_reindex_methods`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_boolean`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_dups`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_columns`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_sparse`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_without_upcasting`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_with_nans`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_axis_style_raises`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_uint_dtypes_fill_value`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_level_verify`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_signature`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_datetimelike_to_object`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_objects`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_fill_value`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_with_duplicate_columns`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_axis_style`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_positional_raises`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_int`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_nan`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_level_verify_repeats`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_axes`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_name_remains`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_api_equivalence`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_empty`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_with_categoricalindex`
- Method removed from `TestDataFrameSelectReindex`: `test_invalid_method`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_multi_categorical_time`
- Method removed from `TestDataFrameSelectReindex`: `test_reindex_multiindex_ffill_added_rows`
- Control flow structure modified:
  - `if_statements`: 5 -> 0 (-5 removed)
  - `for_loops`: 9 -> 0 (-9 removed)
  - `with_statements`: 24 -> 0 (-24 removed)

### test_set_index_3c0de8.py

**Changes detected**: class_missing_in_typed, method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `Thing`
- Class removed: `TestSetIndexInvalid`
- Class removed: `TestSetIndexCustomLabelType`
- Method removed from `TestSetIndex`: `test_set_index_period`
- Control flow structure modified:
  - `with_statements`: 18 -> 2 (-16 removed)

### test_to_numeric_199e51.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- Control flow structure modified:
  - `if_statements`: 16 -> 11 (-5 removed)
  - `with_statements`: 14 -> 11 (-3 removed)

### streams_d2f3a4.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `DataQueue`: `feed_eof`
- Method removed from `DataQueue`: `__aiter__`
- Control flow structure modified:
  - `if_statements`: 58 -> 53 (-5 removed)
  - `try_except`: 7 -> 6 (-1 removed)

### test_hashtable_89c8a4.py

**Changes detected**: class_missing_in_typed, parameter_mismatch, control_flow_mismatch

#### What Changed:

- Class removed: `TestHelpFunctions`
- Class removed: `TestHelpFunctionsWithNans`
- Class removed: `TestHashTableWithNans`
- Function signature changed: `test_map_locations`
  - Original params: `self, table_type, dtype`
  - Modified params: `self, table_type, dtype, writable`
- Control flow structure modified:
  - `if_statements`: 10 -> 8 (-2 removed)
  - `with_statements`: 15 -> 12 (-3 removed)

### test_stata_db4b21.py

**Changes detected**: method_removed_in_typed, control_flow_mismatch

#### What Changed:

- Method removed from `TestStata`: `test_big_dates`
- Method removed from `TestStata`: `test_write_missing_strings`
- Method removed from `TestStata`: `test_stata_111`
- Method removed from `TestStata`: `test_gzip_writing`
- Method removed from `TestStata`: `test_write_variable_labels`
- Method removed from `TestStata`: `test_read_write_reread_dta15`
- Method removed from `TestStata`: `test_read_write_dta13`
- Method removed from `TestStata`: `test_invalid_date_conversion`
- Method removed from `TestStata`: `test_iterator`
- Method removed from `TestStata`: `test_inf`
- Method removed from `TestStata`: `test_path_pathlib`
- Method removed from `TestStata`: `test_value_labels_old_format`
- Method removed from `TestStata`: `test_read_chunks_117`
- Method removed from `TestStata`: `test_mixed_string_strl`
- Method removed from `TestStata`: `test_set_index`
- Method removed from `TestStata`: `test_categorical_with_stata_missing_values`
- Method removed from `TestStata`: `test_numeric_column_names`
- Method removed from `TestStata`: `test_read_data_int_validranges_compat`
- Method removed from `TestStata`: `test_stata_doc_examples`
- Method removed from `TestStata`: `test_encoding`
- Method removed from `TestStata`: `test_categorical_writing`
- Method removed from `TestStata`: `test_read_chunks_115`
- Method removed from `TestStata`: `test_read_write_dta12`
- Method removed from `TestStata`: `test_invalid_file_not_written`
- Method removed from `TestStata`: `test_read_data_int_validranges`
- Method removed from `TestStata`: `test_read_chunks_columns`
- Method removed from `TestStata`: `test_missing_value_conversion_compat`
- Method removed from `TestStata`: `test_105`
- Method removed from `TestStata`: `test_stata_119`
- Method removed from `TestStata`: `test_repeated_column_labels`
- Method removed from `TestStata`: `test_categorical_order`
- Method removed from `TestStata`: `test_missing_value_conversion`
- Method removed from `TestStata`: `_convert_categorical`
- Method removed from `TestStata`: `test_read_write_dta11`
- Method removed from `TestStata`: `test_out_of_range_float`
- Method removed from `TestStata`: `test_invalid_timestamp`
- Method removed from `TestStata`: `test_write_preserves_original`
- Method removed from `TestStata`: `test_invalid_variable_label_encoding`
- Method removed from `TestStata`: `test_nonfile_writing`
- Method removed from `TestStata`: `test_date_parsing_ignores_format_details`
- Method removed from `TestStata`: `test_unicode_dta_118_119`
- Method removed from `TestStata`: `test_read_data_int_validranges_compat_nobyte`
- Method removed from `TestStata`: `test_all_none_exception`
- Method removed from `TestStata`: `test_write_variable_label_errors`
- Method removed from `TestStata`: `test_categorical_ordering`
- Method removed from `TestStata`: `test_drop_column`
- Method removed from `TestStata`: `test_invalid_variable_labels`
- Method removed from `TestStata`: `test_strl_latin1`
- Method removed from `TestStata`: `test_minimal_size_col`
- Method removed from `TestStata`: `test_writer_118_exceptions`
- Method removed from `TestStata`: `test_timestamp_and_label`
- Method removed from `TestStata`: `test_no_index`
- Method removed from `TestStata`: `test_value_labels_iterator`
- Method removed from `TestStata`: `test_utf8_writer`
- Method removed from `TestStata`: `test_string_no_dates`
- Method removed from `TestStata`: `test_convert_strl_name_swap`
- Method removed from `TestStata`: `test_variable_labels`
- Method removed from `TestStata`: `test_default_date_conversion`
- Method removed from `TestStata`: `test_missing_value_conversion_compat_nobyte`
- Method removed from `TestStata`: `test_categorical_warnings_and_errors`
- Method removed from `TestStata`: `test_bool_uint`
- Method removed from `TestStata`: `test_writer_117`
- Method removed from `TestStata`: `test_missing_value_generator`
- Method removed from `TestStata`: `test_read_write_reread_dta14`
- Method removed from `TestStata`: `test_dtype_conversion`
- Method removed from `TestStata`: `test_nan_to_missing_value`
- Method removed from `TestStata`: `test_large_value_conversion`
- Method removed from `TestStata`: `test_excessively_long_string`
- Method removed from `TestStata`: `test_unsupported_datetype`
- Method removed from `TestStata`: `test_out_of_range_double`
- Method removed from `TestStata`: `test_date_export_formats`
- Method removed from `TestStata`: `test_read_write_ea_dtypes`
- Method removed from `TestStata`: `test_categorical_sorting`
- Method removed from `TestStata`: `test_dates_invalid_column`
- Method removed from `TestStata`: `test_unsupported_type`
- Method removed from `TestStata`: `test_encoding_latin1_118`
- Control flow structure modified:
  - `if_statements`: 31 -> 4 (-27 removed)
  - `for_loops`: 28 -> 3 (-25 removed)
  - `try_except`: 2 -> 0 (-2 removed)
  - `with_statements`: 86 -> 4 (-82 removed)

### test_to_html_24b1e4.py

**Changes detected**: class_missing_in_typed, control_flow_mismatch

#### What Changed:

- Class removed: `TestHTMLIndex`
- Class removed: `TestReprHTML`
- Control flow structure modified:
  - `if_statements`: 16 -> 5 (-11 removed)
  - `for_loops`: 4 -> 1 (-3 removed)
  - `with_statements`: 29 -> 8 (-21 removed)

