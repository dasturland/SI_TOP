# University Student ML Lab Pro

**Universitet talabalarini tasniflash va klasterlash** (interaktiv mashinaviy o'rganish laboratoriyasi)

- **Muallif:** Sadullayev Jaxongir
- **Fan:** Sun'iy intellekt texnologiyalari
- **Ish turi:** Laboratoriya ishi
- **GitHub:** https://github.com/dasturland/SI_TOP.git

## Maqsad

Do'kon mijozlarini tasniflash va klasterlash topshirig'i universitet talabalariga moslashtirilgan: mijoz → talaba, xarid summasi → o'rtacha akademik ball, xaridlar soni → darsga qatnashish va topshiriqlarni bajarish ko'rsatkichi. Loyiha uch algoritmni noldan, kutubxonasiz amalga oshiradi: **KNN (k = 3)**, **Qaror daraxti (Gini, maxDepth = 2)** va **K-means (q = 3)**.

> Cheklov: o'qituvchining namuna sayti va PDF topshirig'i loyiha yaratilayotganda tekshirilmadi. Dizayn nusxalanmagan, topshiriq matni asosida mustaqil yaratilgan.

## Texnologiyalar

Faqat HTML5, CSS3, Vanilla JavaScript (ES6+), Canvas va SVG. Framework, kutubxona, tashqi CDN, shrift, backend va build jarayoni yo'q. Internetsiz ishlaydi.

## Dataset

20 talaba (o'quv maqsadidagi namunaviy ma'lumot): T1–T20, 13 ta **Oddiy**, 7 ta **Faol**. Ball: min 40, max 96. Qatnashish: min 45, max 96. Sinf yorliqlari faqat KNN va qaror daraxti uchun ishlatiladi; K-means ularga tayanmaydi.

## Formulalar

- **Min-Max:** `z = (x - min) / (max - min)`; `max = min` bo'lsa `z = 0`. Min/max faqat 20 o'quv talabadan olinadi. Yangi talabaning diapazondan tashqari qiymati kesilmaydi (manfiy bo'lishi mumkin).
- **Evklid masofasi:** `d = sqrt((a1-b1)^2 + (a2-b2)^2)`.
- **Gini:** `G = 1 - Σ p_c^2`; **vaznli Gini:** `G_split = (nL/n)·GL + (nR/n)·GR`.
- **K-means markazi:** `μ_r = (1/|C_r|) Σ z_i`; **inertia:** `J = Σ ||z_i - μ_c(i)||^2` (yangilangan markazlar bilan).

## Algoritmlar

- **KNN:** yangi talaba normalizatsiya qilinadi, 20 talabagacha masofa hisoblanadi, eng yaqin 3 tasi ovoz beradi (teng masofada ID bo'yicha tartib).
- **Qaror daraxti:** har tugunda o'rta nuqta nomzodlari sinab ko'riladi, eng kichik vaznli Gini tanlanadi. Natija: ildiz `Ball <= 70.5`, o'ng tarmoq `Qatnashish <= 74.5`.
- **K-means:** boshlang'ich markazlar C1 = T1, C2 = T15, C3 = T20. Biriktirish, markazni yangilash, J, konvergensiya (yoki 100 iteratsiya). Bo'sh klasterda eski markaz saqlanadi.

## Fayllar tuzilishi

```
index.html        sahifa tuzilishi
style.css         dizayn tizimi (Dark/Light, responsiv, 3D)
script.js         ma'lumotlar, ML algoritmlari, tekshirish, holat, UI
tests/lab.test.js Node testlari
README.md  .gitignore
```

## VS Code'da ishlatish

1. Repozitoriyani klonlang: `git clone https://github.com/dasturland/SI_TOP.git`
2. VS Code'da **File → Open Folder** orqali `SI_TOP` papkasini oching.
3. `index.html` faylini brauzerda oching (server shart emas).
4. Ixtiyoriy mahalliy server: `python -m http.server 5500`, so'ng `http://localhost:5500` manziliga kiring.

## Testlar

Node.js 18+ kerak, paket o'rnatish shart emas:

```
node --check script.js
node --test tests/lab.test.js
```

(`node --test` bevosita ham `tests/` papkasidagi testlarni topadi. Node 22 da `node --test tests/` papka yo'li bilan ishlamasligi mumkin.)

## Muammolarni bartaraf etish

- Sahifa bo'sh ko'rinsa, brauzer konsolini (F12) oching va `script.js` yuklanganini tekshiring.
- Nusxalash ishlamasa, brauzer clipboard ruxsatini tekshiring; dastur muqobil usulni avtomatik sinaydi.
- Tema yoki grafik eskirgan bo'lsa, sahifani yangilang (Ctrl+F5).

## GitHub bilan ishlash

```
git add .
git commit -m "feat: ML Lab Pro"
git push origin main
```

Repozitoriya: https://github.com/dasturland/SI_TOP.git

© 2026 Sadullayev Jaxongir. Barcha huquqlar himoyalangan.
