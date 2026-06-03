# دليل إعادة الإنتاج + محتويات حزمة Zenodo

## بيانات الإدخال (Kaggle Dataset)
- StoppingPower.csv (64,612 قياسًا، IAEA-مشتقّ)
- StoppingPower_refs.csv (المراجع)
- target_composition_table.csv (جدول تركيب الأهداف، مبنيّ داخليًا)

## خطوات التنفيذ (خلايا Kaggle، بالترتيب)
1. STEP 1  — تحميل/تنظيف/توحيد وحدات/دمج التركيب/واصفات/تقسيم → features_full.parquet
2. STEP 2  — خط أساس GBM + مجموعة عميقة → results_step2.json, fig_step2.png
3. STEP 3b — BNN تغايري (واصفات مهذّبة) → results_step3b.json
4. STEP 4  — معايرة حرارية → results_step4.json
5. STEP 5b — استخراج الأُس α(E) → alpha_vs_energy.csv, fig_step5b_alpha.png
6. STEP 6  — اختبار التبادلية → results_step6.json, fig_step6_reciprocity.png
7. STEP 7b — مقارنة ESPNN → results_step7.json
8. STEP 9  — BNN بايزي جزئيًا → pbnn_weights.pt
9. STEP 10 — NGBoost (دقّة + معايرة ذاتية) → ngboost_model.pkl
+ شكل الـ BNN المُعاير → fig_bnn_calibrated.png

## مخرجات للرفع على Zenodo (من /kaggle/working)
features_full.parquet · target_composition_table.csv · bnn_weights.pt · pbnn_weights.pt ·
ngboost_model.pkl · results_step*.json · alpha_vs_energy.csv ·
fig_bnn_calibrated.png · fig_step5b_alpha.png · fig_step6_reciprocity.png · fig_step2.png ·
كل سكربتات الخلايا (STEP 1..10)

## بيئة
Python 3.12 · torch (GPU) · scikit-learn · ngboost · scipy · pandas/numpy · ESPNN==1.0.1 (للمقارنة فقط، تثبيت --no-deps)
