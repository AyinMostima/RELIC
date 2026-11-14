# L2 Module Integration Tests

## 姒傝堪

鏈洰褰曞寘鍚玆ELIC绯荤粺鐨凩2锛堟ā鍧楅棿闆嗘垚锛夋祴璇曘€傝繖浜涙祴璇曢獙璇佷笉鍚岀瀛︽ā鍧椾箣闂寸殑闆嗘垚锛岀‘淇濈郴缁熺殑鏁翠綋鍗忚皟鎬у拰绉戝鍑嗙‘鎬с€?

## 娴嬭瘯鏂囦欢

### test_module_integration.py

瀹屾暣鐨凩2闆嗘垚娴嬭瘯濂椾欢锛屽寘鍚互涓嬫祴璇曠被鍒細

#### 1. 鐘舵€佺鐞嗗櫒闆嗘垚娴嬭瘯 (TestStateManagerIntegration)
- **娴嬭瘯鐩爣**: 楠岃瘉鐘舵€佺鐞嗗櫒鐨勫熀纭€鍔熻兘
- **鍏抽敭娴嬭瘯**:
  - 鐘舵€佽缃拰鑾峰彇鎿嶄綔
  - 鐘舵€佹洿鏂板簲鐢?
  - 鏁版嵁瀹屾暣鎬т繚璇?

#### 2. 鍙傛暟绯荤粺闆嗘垚娴嬭瘯 (TestParameterSystemIntegration)
- **娴嬭瘯鐩爣**: 楠岃瘉鍙傛暟绠＄悊绯荤粺涓庢ā鍧楃殑闆嗘垚
- **鍏抽敭娴嬭瘯**:
  - 鍙傛暟瀛樺偍鍜屾绱?
  - 鍛藉悕绌洪棿缁勭粐
  - 璺ㄦā鍧楀弬鏁拌闂?

#### 3. 绉戝妯″潡闆嗘垚娴嬭瘯 (TestScientificModuleIntegration)
- **娴嬭瘯鐩爣**: 楠岃瘉绉戝妯″潡闂寸殑鑰﹀悎鍜屼氦浜?
- **鍏抽敭娴嬭瘯**:
  - 鍏夊悎浣滅敤-鍛煎惛浣滅敤鑰﹀悎
  - 纰?姘惊鐜€﹀悎
  - 姘?纰宠€﹀悎閫氳繃姘斿瓟璋冭妭

#### 4. 瀹堟亽鍘熺悊娴嬭瘯 (TestConservationPrinciples)
- **娴嬭瘯鐩爣**: 楠岃瘉鐗╃悊瀹堟亽瀹氬緥鐨勯伒瀹?
- **鍏抽敭娴嬭瘯**:
  - 璐ㄩ噺瀹堟亽锛堢敓鐗╁湴鐞冨寲瀛﹀惊鐜級
  - 鑳介噺瀹堟亽锛堣〃闈㈣兘閲忓钩琛★級
  - 姘村钩琛″畧鎭?

#### 5. 澶氭ā鍧楁墽琛屾祴璇?(TestMultiModuleExecution)
- **娴嬭瘯鐩爣**: 楠岃瘉澶氫釜妯″潡鐨勫崗璋冩墽琛?
- **鍏抽敭娴嬭瘯**:
  - 渚濊禆閾炬墽琛?
  - 骞惰鎵ц缁?

## 杩愯娴嬭瘯

### 浣跨敤pytest杩愯
```bash
cd F:\Homework\GEOBIOCHE\RELIC-REBUILD
python -m pytest test/integration/test_module_integration.py -v
```

### 鐩存帴杩愯娴嬭瘯鏂囦欢
```bash
cd F:\Homework\GEOBIOCHE\RELIC-REBUILD
python test/integration/test_module_integration.py
```

## 娴嬭瘯缁撴灉瑙ｈ

### 鎴愬姛鏍囧噯
- **鎬讳綋鎴愬姛鐜?*: 鈮?0% 琚涓烘槸鍙帴鍙楃殑
- **鍏抽敭娴嬭瘯**: 瀹堟亽鍘熺悊娴嬭瘯蹇呴』閫氳繃
- **璺宠繃娴嬭瘯**: 鐢变簬妯″潡鍙敤鎬у鑷寸殑璺宠繃鏄甯哥殑

### 褰撳墠娴嬭瘯鐘舵€?
鏈€鍚庝竴娆¤繍琛岀粨鏋滐細
- **鎬绘祴璇曟暟**: 12
- **澶辫触**: 1 
- **閿欒**: 0
- **璺宠繃**: 2
- **鎴愬姛鐜?*: 91.7%

## 妯″潡渚濊禆

娴嬭瘯浼氳嚜鍔ㄦ娴嬩互涓嬫ā鍧楃殑鍙敤鎬э細
- [OK] 鏍稿績妯″潡 (Core modules)
- [OK] Functor鍩虹被 (Functor base)
- [OK] 妯℃嫙鍣?(Simulator)
- [OK] 渚濊禆鍥?(Dependency graph)

涓嶅彲鐢ㄧ殑妯″潡浼氬鑷寸浉鍏虫祴璇曡璺宠繃锛岃繖鏄甯歌涓恒€?

## 绉戝楠岃瘉

### 娴嬭瘯鐨勭瀛﹀師鐞?

1. **澶ф皵-鐢熺墿鍦堣€﹀悎**
   - 姘斿瓟瀵煎害妯″瀷: g_s = g_0 + g_1 * A_n * h / (C_s - 螕*)
   - 鍏夊悎浣滅敤妯″瀷: A_n = min(A_c, A_j) - R_d
   - 鑳介噺骞宠　: R_n = H + 位E + G

2. **姘存枃-鍦熷￥鑰﹀悎**
   - Richards鏂圭▼: 鈭偽?鈭倀 = 鈭?鈭倆[K(胃)(鈭傁?鈭倆 + 1)]
   - 鏍圭郴鍚告按: S = 伪(蠄) * R(z) * T_p

3. **纰冲惊鐜畧鎭?*
   - 鍏ㄧ悆纰冲钩琛? dC_total/dt = 0
   - 鍖哄骞宠　: dC_i/dt = 危(inputs) - 危(outputs)

### 鍙傝€冩枃鐚?
- Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem Modeling
- Fisher, R.A. et al. (2018). Vegetation demographics in Earth System Models
- Lawrence, D.M. et al. (2019). The Community Land Model version 5

## 鏁呴殰鎺掗櫎

### 甯歌闂

1. **Unicode缂栫爜閿欒**
   - 宸插湪浠ｇ爜涓敤ASCII瀛楃鏇挎崲鐗规畩绗﹀彿

2. **妯″潡瀵煎叆澶辫触**
   - 娴嬭瘯鍖呭惈浼橀泤鐨刦allback鏈哄埗
   - 涓嶅彲鐢ㄦā鍧椾細琚烦杩囪€屼笉浼氬鑷存祴璇曞け璐?

3. **鏁板€肩簿搴﹂棶棰?*
   - 瀹堟亽娴嬭瘯浣跨敤閫傚綋鐨勫宸?
   - PyTorch璁剧疆涓哄弻绮惧害妯″紡

### 璋冭瘯鎶€宸?

1. 杩愯鍗曚釜娴嬭瘯绫伙細
```bash
python -m pytest test/integration/test_module_integration.py::TestStateManagerIntegration -v
```

2. 鏌ョ湅璇︾粏鏃ュ織锛?
```bash
python test/integration/test_module_integration.py
```

3. 妫€鏌ユā鍧楀彲鐢ㄦ€э細
娴嬭瘯寮€濮嬫椂浼氭樉绀哄悇妯″潡鐨勫彲鐢ㄦ€х姸鎬併€?

## 鎵╁睍娴嬭瘯

瑕佹坊鍔犳柊鐨勯泦鎴愭祴璇曪細

1. 鍒涘缓鏂扮殑娴嬭瘯绫伙紝缁ф壙`unittest.TestCase`
2. 鍦╜run_comprehensive_l2_tests()`鍑芥暟涓坊鍔犳祴璇曠被
3. 閬靛惊鐜版湁鐨勫懡鍚嶇害瀹氬拰鏂囨。鏍煎紡
4. 纭繚鍖呭惈閫傚綋鐨勭瀛﹂獙璇?

## 缁存姢璇存槑

- 瀹氭湡鏇存柊娴嬭瘯浠ュ弽鏄犳柊鐨勭瀛︽ā鍧?
- 淇濇寔瀹堟亽娴嬭瘯鐨勬暟鍊煎钩琛?
- 鏇存柊鏂囨。涓殑鍙傝€冩枃鐚?
- 鐩戞帶娴嬭瘯鎴愬姛鐜囪秼鍔
