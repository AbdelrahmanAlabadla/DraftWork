// UI language layer. Independent from the exam/document language: the UI can
// be Arabic while generated exams follow the uploaded document's language.
const STRINGS = {
  en: {
    "header.tag": "RAG-POWERED",
    "nav.eval": "Eval Dashboard",
    "auth.connect": "Connect Google Account",
    "auth.signout": "Sign out",
    "section.upload": "01 — Upload Document",
    "upload.title": "Drop your file here",
    "upload.hint": "Click to browse or drag and drop your document",
    "section.settings": "02 — Exam Settings",
    "settings.models": "Number of Models",
    "settings.difficulty": "Difficulty",
    "diff.easy": "Easy",
    "diff.medium": "Medium",
    "diff.hard": "Hard",
    "diff.mix": "Mix",
    "details.title": "Printed Exam Details",
    "details.optional": "optional",
    "field.exam_title": "Exam title",
    "field.exam_title_ph": "Examination",
    "field.class": "Class",
    "field.class_ph": "e.g. Grade 12",
    "field.duration": "Duration",
    "field.duration_ph": "e.g. 90 minutes",
    "field.exam_date": "Exam date",
    "field.teacher": "Teacher",
    "field.teacher_ph": "Teacher name",
    "field.left_logo": "Left logo",
    "field.right_logo": "Right logo",
    "details.note": "PDF and DOCX use A4 pages with 0.5-inch margins on every side. Logos must be under 2 MB.",
    "section.sections": "03 — Choose Sections To Include In The Exam",
    "sections.loading": "Generating sections, your titles will appear in a few seconds...",
    "sections.select_all": "Select All Titles",
    "section.qtypes": "04 — Question Types",
    "qtypes.total": "Total: {n} questions",
    "qtype.mcq": "Multiple Choice",
    "qtype.mcq_desc": "4 options, single answer",
    "qtype.tf": "True / False",
    "qtype.tf_desc": "Quick recall checks",
    "qtype.fitb": "Fill in the Blank",
    "qtype.fitb_desc": "Key term recall",
    "qtype.why": "Why Questions",
    "qtype.why_desc": "Short reasoning",
    "qtype.essay": "Essay",
    "qtype.essay_desc": "Open-ended responses",
    "gen.status": "Generating exam — this may take a moment...",
    "gen.button": "Generate Exam →",
    "preview.title": "Exam Preview",
    "preview.copy_md": "Copy Markdown",
    "preview.export": "Export ▾",
    "alert.upload_first": "Please upload and index a file first.",
    "alert.select_sections": "Please select at least one subsection to include in the exam.",
  },
  ar: {
    "header.tag": "مدعوم بالاسترجاع الذكي",
    "nav.eval": "لوحة التقييم",
    "auth.connect": "ربط حساب Google",
    "auth.signout": "تسجيل الخروج",
    "section.upload": "٠١ — رفع المستند",
    "upload.title": "أفلت ملفك هنا",
    "upload.hint": "انقر للاختيار أو اسحب وأفلت مستندك",
    "section.settings": "٠٢ — إعدادات الامتحان",
    "settings.models": "عدد النسخ",
    "settings.difficulty": "مستوى الصعوبة",
    "diff.easy": "سهل",
    "diff.medium": "متوسط",
    "diff.hard": "صعب",
    "diff.mix": "متنوع",
    "details.title": "تفاصيل الامتحان المطبوع",
    "details.optional": "اختياري",
    "field.exam_title": "عنوان الامتحان",
    "field.exam_title_ph": "امتحان",
    "field.class": "الصف",
    "field.class_ph": "مثال: الصف الثاني عشر",
    "field.duration": "المدة",
    "field.duration_ph": "مثال: ٩٠ دقيقة",
    "field.exam_date": "تاريخ الامتحان",
    "field.teacher": "المعلم",
    "field.teacher_ph": "اسم المعلم",
    "field.left_logo": "الشعار الأيسر",
    "field.right_logo": "الشعار الأيمن",
    "details.note": "تستخدم ملفات PDF و DOCX صفحات A4 بهوامش نصف بوصة من كل الجوانب. يجب أن يكون حجم الشعار أقل من ٢ ميغابايت.",
    "section.sections": "٠٣ — اختر الأقسام لتضمينها في الامتحان",
    "sections.loading": "جارٍ إنشاء الأقسام، ستظهر العناوين خلال ثوانٍ...",
    "sections.select_all": "تحديد كل العناوين",
    "section.qtypes": "٠٤ — أنواع الأسئلة",
    "qtypes.total": "الإجمالي: {n} سؤالاً",
    "qtype.mcq": "اختيار من متعدد",
    "qtype.mcq_desc": "٤ خيارات، إجابة واحدة",
    "qtype.tf": "صح / خطأ",
    "qtype.tf_desc": "تحقق سريع من التذكر",
    "qtype.fitb": "أكمل الفراغ",
    "qtype.fitb_desc": "تذكر المصطلحات الأساسية",
    "qtype.why": "أسئلة التحليل",
    "qtype.why_desc": "استنتاج قصير",
    "qtype.essay": "مقالي",
    "qtype.essay_desc": "إجابات مفتوحة",
    "gen.status": "جارٍ إنشاء الامتحان — قد يستغرق بعض الوقت...",
    "gen.button": "إنشاء الامتحان ←",
    "preview.title": "معاينة الامتحان",
    "preview.copy_md": "نسخ Markdown",
    "preview.export": "تصدير ▾",
    "alert.upload_first": "يرجى رفع ملف وفهرسته أولاً.",
    "alert.select_sections": "يرجى اختيار قسم فرعي واحد على الأقل لتضمينه في الامتحان.",
  },
};

let current = localStorage.getItem("ui_lang") || "en";

export function uiLang() {
  return current;
}

export function t(key, params) {
  let text = (STRINGS[current] && STRINGS[current][key]) || STRINGS.en[key] || key;
  if (params) {
    for (const [name, value] of Object.entries(params)) {
      text = text.replace(`{${name}}`, String(value));
    }
  }
  return text;
}

export function applyUiLanguage(lang) {
  if (!STRINGS[lang]) return;
  current = lang;
  localStorage.setItem("ui_lang", lang);
  const rtl = lang === "ar";
  document.documentElement.lang = lang;
  document.documentElement.dir = rtl ? "rtl" : "ltr";
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of document.querySelectorAll("[data-i18n-ph]")) {
    el.placeholder = t(el.dataset.i18nPh);
  }
  const switcher = document.getElementById("langSwitch");
  if (switcher) switcher.textContent = rtl ? "EN" : "ع";
}

export function initI18n() {
  const header = document.querySelector(".header");
  if (header && !document.getElementById("langSwitch")) {
    const btn = document.createElement("button");
    btn.id = "langSwitch";
    btn.className = "copy-btn";
    btn.style.cssText = "margin-inline-start:auto;";
    btn.addEventListener("click", () =>
      applyUiLanguage(current === "en" ? "ar" : "en")
    );
    header.appendChild(btn);
  }
  applyUiLanguage(current);
}
