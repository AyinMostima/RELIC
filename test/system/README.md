# RELIC L3 System Tests

杩欎釜鐩綍鍖呭惈RELIC鐢熸€佺郴缁熸ā鍨嬬殑L3绾э紙绯荤粺绾э級绔埌绔祴璇曘€傝繖浜涙祴璇曢獙璇佹暣涓缓妯℃鏋剁殑鍔熻兘銆佹€ц兘鍜岀瀛﹀噯纭€с€?

## 娴嬭瘯姒傝堪

### 涓昏娴嬭瘯鏂囦欢

- `test_end_to_end.py` - 涓昏鐨凩3绯荤粺娴嬭瘯濂椾欢
- `run_l3_tests.py` - 娴嬭瘯杩愯鍣ㄨ剼鏈?
- `README.md` - 姝ゆ枃妗?

### 娴嬭瘯瑕嗙洊鑼冨洿

L3绯荤粺娴嬭瘯鍖呭惈浠ヤ笅鍏抽敭娴嬭瘯鍦烘櫙锛?

1. **瀹屾暣鏃ュ惊鐜ā鎷?* (`test_complete_daily_cycle_simulation`)
   - 24灏忔椂妯℃嫙锛屾瘡灏忔椂鏃堕棿姝ラ暱
   - 楠岃瘉鏄煎鑺傚緥妯″紡锛堝厜鍚堜綔鐢ㄣ€佽捀鏁ｅ彂绛夛級
   - 鑳介噺鍜岃川閲忓钩琛℃楠?
   - 绯荤粺鍦ㄦ棩寰幆涓殑绋冲畾鎬?

2. **骞村害鐢熸€佺郴缁熸ā鎷?* (`test_annual_ecosystem_simulation`)
   - 365澶╂ā鎷燂紝鏃ユ椂闂存闀?
   - 瀛ｈ妭鎬фā寮忛獙璇?
   - 澶氱敓鎬佺郴缁熺被鍨嬫瘮杈冿紙娓╁甫妫灄銆佽崏鍦般€佸啘鐢帮級
   - 骞村害纰冲钩琛″垎鏋?

3. **澶氱珯鐐规瘮杈冩祴璇?* (`test_multi_site_comparison`)
   - 鍚屾椂妯℃嫙澶氫釜绔欑偣
   - 绔欑偣鐗瑰紓鎬у弬鏁板彉鍖?
   - 璁＄畻鏁堢巼鍜屽彲鎵╁睍鎬ф祴璇?
   - 绔欑偣闂村樊寮傜殑鐜板疄鎬ч獙璇?

4. **鏋佺鏉′欢鍘嬪姏娴嬭瘯** (`test_extreme_conditions_stress_testing`)
   - 骞叉棻鑳佽揩鍝嶅簲锛?0澶╂棤闄嶆按锛?
   - 鐑氮鏉′欢锛?40掳C鎸佺画10澶╋級
   - 瀵掓疆鏉′欢锛?-10掳C鎸佺画10澶╋級
   - 娲稘鍦烘櫙锛堣繃閲忛檷姘达級
   - 鏋佺浜嬩欢鍚庣殑鎭㈠鑳藉姏

5. **鎬ц兘鍩哄噯娴嬭瘯** (`test_performance_benchmarking`)
   - 鎵ц鏃堕棿闅忛棶棰樿妯＄殑鎵╁睍鎬?
   - 鍐呭瓨浣跨敤妯″紡鍒嗘瀽
   - CPU鍒╃敤鐜囨晥鐜?
   - GPU鍔犻€熸瘮杈冿紙濡傛灉鍙敤锛?
   - 璁＄畻鐡堕璇嗗埆

6. **绯荤粺椴佹鎬у拰鎭㈠娴嬭瘯** (`test_system_robustness_and_recovery`)
   - 鏃犳晥杈撳叆鐨勪紭闆呭鐞?
   - 鏁板€间笉绋冲畾鎬ф仮澶?
   - 閿欒浼犳挱鎺у埗
   - 鍥為€€鏈哄埗鏈夋晥鎬?
   - 閿欒鍚庣殑鐘舵€佷竴鑷存€?

7. **绉戝鍑嗙‘鎬ч獙璇?* (`test_scientific_accuracy_validation`)
   - 鍏夊悎浣滅敤閫熺巼 vs Farquhar妯″瀷鏈熸湜鍊?
   - 钂告暎鍙?vs Penman-Monteith鍩哄噯
   - 鍛煎惛浣滅敤娓╁害渚濊禆鎬э紙Q10鍏崇郴锛?
   - 纰冲埄鐢ㄦ晥鐜囧€?
   - 姘村垎鍒╃敤鏁堢巼鑼冨洿
   - 鑳介噺骞宠　闂悎

## 浣跨敤鏂规硶

### 鐜瑕佹眰

鍦ㄨ繍琛孡3娴嬭瘯涔嬪墠锛岀‘淇濆凡瀹夎浠ヤ笅渚濊禆锛?

```bash
# 鏍稿績渚濊禆
pip install torch numpy matplotlib psutil pytest

# 鍙€変緷璧栵紙鐢ㄤ簬澧炲己鍔熻兘锛?
pip install plotly seaborn jupyter
```

### 杩愯鎵€鏈夋祴璇?

```bash
# 杩愯瀹屾暣鐨凩3娴嬭瘯濂椾欢
python run_l3_tests.py

# 杩愯璇︾粏杈撳嚭
python run_l3_tests.py --verbose

# 淇濆瓨璇婃柇鍥捐〃
python run_l3_tests.py --save-plots
```

### 杩愯鐗瑰畾娴嬭瘯

```bash
# 杩愯鏃ュ惊鐜祴璇?
python run_l3_tests.py test_complete_daily_cycle_simulation

# 杩愯骞村害妯℃嫙娴嬭瘯
python run_l3_tests.py test_annual_ecosystem_simulation

# 杩愯鎬ц兘鍩哄噯娴嬭瘯
python run_l3_tests.py test_performance_benchmarking
```

### 浣跨敤pytest鐩存帴杩愯

```bash
# 杩愯鎵€鏈塋3娴嬭瘯
pytest test_end_to_end.py -v

# 杩愯鐗瑰畾娴嬭瘯绫?
pytest test_end_to_end.py::TestL3EndToEndSystem -v

# 杩愯鐗瑰畾娴嬭瘯鏂规硶
pytest test_end_to_end.py::TestL3EndToEndSystem::test_complete_daily_cycle_simulation -v
```

### 杈撳嚭鍜岀粨鏋?

娴嬭瘯杩愯鍚庯紝浼氱敓鎴愪互涓嬭緭鍑猴細

1. **娴嬭瘯鎶ュ憡**
   - JSON鏍煎紡鐨勮缁嗘姤鍛?(`l3_test_report_YYYYMMDD_HHMMSS.json`)
   - 鏂囨湰鏍煎紡鐨勬憳瑕?(`l3_test_summary.txt`)

2. **绉戝鏁版嵁鏂囦欢**
   - 鏃ュ惊鐜粨鏋?(`daily_cycle_results.json`)
   - 骞村害妯℃嫙缁撴灉 (`annual_simulation_results.json`)
   - 澶氱珯鐐规瘮杈冪粨鏋?(`multi_site_results.json`)
   - 鏋佺鏉′欢缁撴灉 (`extreme_conditions_results.json`)
   - 鎬ц兘鍩哄噯缁撴灉 (`performance_benchmark_results.json`)

3. **璇婃柇鍥捐〃**锛堜娇鐢?`--save-plots`锛?
   - 绯荤粺娴嬭瘯璇婃柇鍥?(`l3_system_test_diagnostics.png`)

## 鎬ц兘鐩爣

L3娴嬭瘯楠岃瘉绯荤粺鏄惁婊¤冻浠ヤ笅鎬ц兘鐩爣锛?

- **鍗曟椂闂存**锛?00涓珯鐐?< 1绉?
- **鍐呭瓨澧為暱**锛?000鏃堕棿姝?< 10MB
- **鍙墿灞曟€?*锛氭椂闂村鏉傚害涓庣珯鐐规暟浜氱嚎鎬у闀?
- **GPU鍔犻€?*锛氬ぇ鎵规澶勭悊鏃?> 0.8鍊嶅姞閫?

## 绉戝楠岃瘉鏍囧噯

### 鍏夊悎浣滅敤
- C3妞嶇墿GPP鑼冨洿锛?-50 渭mol CO2 m鈦宦?s鈦宦?
- 鍏夊搷搴旓細浣庡厜鏉′欢涓婫PP搴旈檷浣?0-90%
- CO2鍝嶅簲锛欳O2鍊嶅鏃禛PP澧炲姞1.0-2.5鍊?

### 钂告暎鍙?
- 鍙傝€僂T鑼冨洿锛?-8 mm/澶╋紙涓嶅悓鐢熸€佺郴缁燂級
- 娓╁害鍝嶅簲锛氱儹骞叉潯浠跺鍔狅紝鍐锋箍鏉′欢鍑忓皯
- 椋庨€熷搷搴旓細椋庨€熷鍔犳椂ET澧炲姞

### 鍛煎惛浣滅敤
- Q10鍊硷細1.5-3.5锛堝吀鍨嬪€肩害2.0锛?
- 娓╁害渚濊禆鎬э細10掳C娓╁崌瀵艰嚧鍛煎惛閫熺巼澧炲姞Q10鍊?

### 鑳介噺骞宠　
- 鑳介噺骞宠　闂悎锛?.6-1.2锛堝吀鍨嬬敓鎬佺郴缁燂級
- 婀嶆祦閫氶噺 = 娼滅儹閫氶噺 + 鎰熺儹閫氶噺

## 鏁呴殰鎺掗櫎

### 甯歌闂

1. **鍐呭瓨涓嶈冻閿欒**
   - 鍑忓皬鎵规澶у皬鎴栨ā鎷熸椂闀?
   - 鍚敤淇濆畧妯″紡锛歚strict_conservation_enabled=False`

2. **CUDA閿欒**
   - 妫€鏌PU鍐呭瓨锛歚torch.cuda.memory_summary()`
   - 鍒囨崲鍒癈PU锛氬湪鐘舵€佷腑浣跨敤`.cpu()`

3. **鏁板€间笉绋冲畾**
   - 妫€鏌ユ瀬绔緭鍏ュ€?
   - 鍚敤鍥為€€妯″潡锛歚enable_bootstrap_modules=True`

4. **娴嬭瘯瓒呮椂**
   - 澧炲姞瓒呮椂闄愬埗
   - 妫€鏌ョ郴缁熻祫婧愪娇鐢ㄦ儏鍐?

### 璋冭瘯妯″紡

```bash
# 杩愯璋冭瘯妯″紡
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from test_end_to_end import TestL3EndToEndSystem
import pytest
pytest.main(['test_end_to_end.py::TestL3EndToEndSystem::test_complete_daily_cycle_simulation', '-v', '-s'])
"
```

## 鎵╁睍娴嬭瘯

### 娣诲姞鏂版祴璇曞満鏅?

1. 鍦╜TestL3EndToEndSystem`绫讳腑鍒涘缓鏂版柟娉?
2. 浣跨敤`performance_monitor`鍜宍ecosystem_data` fixtures
3. 閬靛惊鐜版湁娴嬭瘯鐨勫懡鍚嶇害瀹氬拰缁撴瀯
4. 娣诲姞閫傚綋鐨勭瀛﹂獙璇佹爣鍑?

### 鑷畾涔夌敓鎬佺郴缁熺被鍨?

淇敼`EcosystemDataGenerator.generate_ecosystem_initial_state()`涓殑`ecosystem_params`瀛楀吀鏉ユ坊鍔犳柊鐨勭敓鎬佺郴缁熺被鍨嬨€?

### 鎬ц兘鍒嗘瀽

```python
# 浣跨敤Python profiler
python -m cProfile -o profile_output.prof run_l3_tests.py test_performance_benchmarking

# 鍒嗘瀽缁撴灉
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

## 璐＄尞鎸囧崡

1. 鎵€鏈夋柊娴嬭瘯蹇呴』鍖呭惈锛?
   - 绉戝鏂囨。鍜屽弬鑰冩枃鐚?
   - 鎬ц兘楠岃瘉鏍囧噯
   - 閿欒澶勭悊娴嬭瘯
   - 閫傚綋鐨勬柇瑷€鍜岄獙璇?

2. 娴嬭瘯搴旇锛?
   - 杩愯鏃堕棿 < 10鍒嗛挓锛堝崟涓祴璇曪級
   - 鍐呭瓨浣跨敤 < 2GB
   - 鎻愪緵娓呮櫚鐨勫け璐ユ秷鎭?
   - 鍖呭惈鐗╃悊鍚堢悊鎬ф鏌?

3. 浠ｇ爜璐ㄩ噺锛?
   - 閬靛惊PEP 8椋庢牸鎸囧崡
   - 鍖呭惈绫诲瀷鎻愮ず
   - 娣诲姞閫傚綋鐨勬敞閲婂拰鏂囨。瀛楃涓?

## 鍙傝€冩枃鐚?

1. Farquhar, G. D., et al. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. Planta, 149(1), 78-90.

2. Monteith, J. L. (1965). Evaporation and environment. Symposia of the Society for Experimental Biology, 19, 205-234.

3. Lloyd, J., & Taylor, J. A. (1994). On the temperature dependence of soil respiration. Functional Ecology, 8(3), 315-323.

4. Campbell, G. S., & Norman, J. M. (1998). An Introduction to Environmental Biophysics. Springer-Verlag.

5. Chapin, F. S., et al. (2011). Principles of Terrestrial Ecosystem Ecology. Springer.
