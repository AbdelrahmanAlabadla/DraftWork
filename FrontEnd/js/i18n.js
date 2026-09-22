// UI language layer. Independent from the exam/document language: the UI can
// be Arabic while generated exams follow the uploaded document's language.
const STRINGS = {
  en: {
    "section.upload": "01 — Upload Document",
    "upload.title": "Drop your file here",
    "upload.hint": "Click to browse or drag and drop your document",
    "upload.stop": "Stop document processing",
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
    "sections.unselect_all": "Unselect All Titles",
    "sections.selected": "{n} selected",
    "sections.title_unavailable": "Section title unavailable",
    "section.qtypes": "04 — Question Types",
    "qtypes.total": "Total: {n} questions",
    "qtype.mcq": "Multiple Choice",
    "qtype.mcq_desc": "4 options, single answer",
    "qtype.tf": "True / False",
    "qtype.tf_desc": "Quick recall checks",
    "qtype.fitb": "Fill in the Blank",
    "qtype.fitb_desc": "Key term recall",
    "qtype.definition": "Definition",
    "qtype.definition_desc": "Define a term or concept",
    "qtype.why": "Why Questions",
    "qtype.why_desc": "Short reasoning",
    "qtype.equation": "Equation",
    "qtype.equation_desc": "Direct calculation with working",
    "qtype.word_problem": "Word Problem",
    "qtype.word_problem_desc": "Interpret a situation and calculate",
    "qtype.essay": "Essay",
    "qtype.essay_desc": "Open-ended responses",
    "gen.status": "Generating exam — this may take a moment...",
    "gen.button": "Generate Exam →",
    "gen.stop": "Stop generation",
    "gen.stopping": "Stopping generation...",
    "preview.title": "Exam Preview",
    "preview.copy_md": "Copy Markdown",
    "preview.export": "Export ▾",
    "toast.copied": "Copied",
    "status.unsupported_pdf": "Only .pdf files are supported.",
    "status.uploading": "Uploading the document",
    "status.file_indexed": "File indexed successfully.",
    "status.load_document_failed": "Could not load the processed document.",
    "status.upload_failed": "Upload failed.",
    "status.server_unreachable": "Cannot reach the server.",
    "status.generation_failed": "Generation failed: {detail}",
    "status.generating_progress": "Generating exam: {stage} ({progress}%)",
    "status.exam_progress": "{stage} ({progress}%)",
    "status.load_exam_failed": "Could not load the generated exam.",
    "status.unknown_error": "Unknown error",
    "status.fastapi_unreachable": "Cannot reach the server. Is FastAPI running?",
    "status.export_requires_exam": "Generate an exam before exporting.",
    "status.select_model": "Select at least one model.",
    "status.creating_export": "Creating {kind} export...",
    "status.export_downloaded": "{kind} ZIP downloaded.",
    "status.export_failed": "Export failed (HTTP {status}).",
    "status.job_failed": "The job failed.",
    "status.job_cancelled": "The job was cancelled.",
    "status.job_read_failed": "Could not read job status.",
    "status.cancel_failed": "Could not stop the job.",
    "status.logo_size": "Each logo must be smaller than 2 MB.",
    "status.logo_type": "Logos must be PNG, JPEG, or WebP images.",
    "status.logo_read": "A logo could not be read.",
    "stage.uploading": "Uploading the document",
    "stage.queued": "Waiting to begin processing",
    "stage.validating": "Checking the document",
    "stage.extracting": "Extracting text and book structure",
    "stage.detecting_language": "Detecting the document language",
    "stage.detecting_headings": "Identifying headings and sections",
    "stage.chunking": "Organizing the book into sections",
    "stage.generating_titles": "Creating descriptive section titles",
    "stage.generating": "Generating",
    "stage.completed": "Document ready",
    "stage.cancelling": "Stopping document processing",
    "stage.cancelled": "Document processing stopped",
    "exam_stage.queued": "Waiting to begin generation",
    "exam_stage.analyzing_request": "Analyzing the request",
    "exam_stage.planning_exam": "Planning the exam",
    "exam_stage.generating_exam": "Generating the exam",
    "exam_stage.validating_exam": "Validating the exam",
    "exam_stage.repairing_exam": "Repairing the exam",
    "exam_stage.validating_updates": "Validating the final updates",
    "exam_stage.completed": "Exam ready",
    "saved.account": "Your account",
    "saved.title": "My Exams",
    "saved.intro": "Open a saved exam directly in the preview.",
    "saved.generated_exam": "Generated exam",
    "saved.models": "{n} models",
    "saved.model": "{n} model",
    "saved.open": "Open exam",
    "saved.opening": "Opening exam...",
    "saved.loading": "Loading your exams...",
    "saved.load_failed": "Could not load your exams.",
    "saved.none": "There is no exam made yet.",
    "saved.try_again": "Please try again.",
    "export.dialog_label": "Export",
    "export.dialog_title": "Select models to export",
    "export.dialog_note": "An answer file is automatically included with every selected model.",
    "export.cancel": "Cancel",
    "export.confirm": "Export selected",
    "export.model": "Model {n}",
    "export.select_all": "Select all",
    "export.clear_all": "Clear all",
    "common.close": "Close",
    "alert.upload_first": "Please upload and index a file first.",
    "alert.select_sections": "Please select at least one subsection to include in the exam.",
  },
  ar: {
    "section.upload": "٠١ — رفع المستند",
    "upload.title": "أفلت ملفك هنا",
    "upload.hint": "انقر للاختيار أو اسحب وأفلت مستندك",
    "upload.stop": "إيقاف معالجة المستند",
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
    "sections.unselect_all": "إلغاء تحديد كل العناوين",
    "sections.selected": "تم تحديد {n}",
    "sections.title_unavailable": "عنوان القسم غير متاح",
    "section.qtypes": "٠٤ — أنواع الأسئلة",
    "qtypes.total": "الإجمالي: {n} سؤالاً",
    "qtype.mcq": "اختيار من متعدد",
    "qtype.mcq_desc": "٤ خيارات، إجابة واحدة",
    "qtype.tf": "صح / خطأ",
    "qtype.tf_desc": "تحقق سريع من التذكر",
    "qtype.fitb": "أكمل الفراغ",
    "qtype.fitb_desc": "تذكر المصطلحات الأساسية",
    "qtype.definition": "تعريف",
    "qtype.definition_desc": "تعريف مصطلح أو مفهوم",
    "qtype.why": "علل",
    "qtype.why_desc": "استنتاج قصير",
    "qtype.equation": "معادلة",
    "qtype.equation_desc": "حساب مباشر مع إظهار الخطوات",
    "qtype.word_problem": "سؤال كلامي",
    "qtype.word_problem_desc": "فهم المسألة وإجراء الحساب",
    "qtype.essay": "مقال",
    "qtype.essay_desc": "إجابات مفتوحة",
    "gen.status": "جارٍ إنشاء الامتحان — قد يستغرق بعض الوقت...",
    "gen.button": "إنشاء الامتحان ←",
    "gen.stop": "إيقاف الإنشاء",
    "gen.stopping": "جارٍ الإيقاف...",
    "preview.title": "معاينة الامتحان",
    "preview.copy_md": "نسخ Markdown",
    "preview.export": "تصدير ▾",
    "toast.copied": "تم النسخ",
    "status.unsupported_pdf": "ملفات PDF فقط مدعومة.",
    "status.uploading": "جارٍ رفع المستند",
    "status.file_indexed": "تمت فهرسة الملف بنجاح.",
    "status.load_document_failed": "تعذر تحميل المستند بعد معالجته.",
    "status.upload_failed": "فشل رفع الملف.",
    "status.server_unreachable": "تعذر الاتصال بالخادم.",
    "status.generation_failed": "فشل إنشاء الامتحان: {detail}",
    "status.generating_progress": "جارٍ إنشاء الامتحان: {stage} ({progress}٪)",
    "status.exam_progress": "{stage} ({progress}٪)",
    "status.load_exam_failed": "تعذر تحميل الامتحان الذي تم إنشاؤه.",
    "status.unknown_error": "خطأ غير معروف",
    "status.fastapi_unreachable": "تعذر الاتصال بالخادم. تحقق من تشغيل FastAPI.",
    "status.export_requires_exam": "أنشئ امتحاناً قبل التصدير.",
    "status.select_model": "اختر نسخة واحدة على الأقل.",
    "status.creating_export": "جارٍ إنشاء ملف {kind}...",
    "status.export_downloaded": "تم تنزيل ملف {kind} المضغوط.",
    "status.export_failed": "فشل التصدير (رمز HTTP ‏{status}).",
    "status.job_failed": "فشلت المهمة.",
    "status.job_cancelled": "أُلغيت المهمة.",
    "status.job_read_failed": "تعذر قراءة حالة المهمة.",
    "status.cancel_failed": "تعذر إيقاف المهمة.",
    "status.logo_size": "يجب أن يكون حجم كل شعار أقل من ٢ ميغابايت.",
    "status.logo_type": "يجب أن تكون الشعارات بصيغة PNG أو JPEG أو WebP.",
    "status.logo_read": "تعذرت قراءة أحد الشعارات.",
    "stage.uploading": "جارٍ رفع المستند",
    "stage.queued": "في انتظار بدء المعالجة",
    "stage.validating": "جارٍ التحقق من المستند",
    "stage.extracting": "جارٍ استخراج النص وبنية الكتاب",
    "stage.detecting_language": "جارٍ تحديد لغة المستند",
    "stage.detecting_headings": "جارٍ تحديد العناوين والأقسام",
    "stage.chunking": "جارٍ تنظيم الكتاب إلى أقسام",
    "stage.generating_titles": "جارٍ إنشاء عناوين وصفية للأقسام",
    "stage.generating": "جارٍ الإنشاء",
    "stage.completed": "المستند جاهز",
    "stage.cancelling": "جارٍ إيقاف معالجة المستند",
    "stage.cancelled": "تم إيقاف معالجة المستند",
    "exam_stage.queued": "في انتظار بدء إنشاء الامتحان",
    "exam_stage.analyzing_request": "جارٍ تحليل الطلب",
    "exam_stage.planning_exam": "جارٍ تخطيط الامتحان",
    "exam_stage.generating_exam": "جارٍ إنشاء الامتحان",
    "exam_stage.validating_exam": "جارٍ التحقق من الامتحان",
    "exam_stage.repairing_exam": "جارٍ إصلاح الامتحان",
    "exam_stage.validating_updates": "جارٍ التحقق من التعديلات النهائية",
    "exam_stage.completed": "الامتحان جاهز",
    "saved.account": "حسابك",
    "saved.title": "امتحاناتي",
    "saved.intro": "افتح امتحاناً محفوظاً مباشرة في المعاينة.",
    "saved.generated_exam": "امتحان مُنشأ",
    "saved.models": "{n} نسخ",
    "saved.model": "نسخة واحدة",
    "saved.open": "فتح الامتحان",
    "saved.opening": "جارٍ فتح الامتحان...",
    "saved.loading": "جارٍ تحميل امتحاناتك...",
    "saved.load_failed": "تعذر تحميل امتحاناتك.",
    "saved.none": "لا توجد امتحانات منشأة بعد.",
    "saved.try_again": "يرجى المحاولة مرة أخرى.",
    "export.dialog_label": "تصدير",
    "export.dialog_title": "اختر النسخ المراد تصديرها",
    "export.dialog_note": "يُرفق ملف الإجابات تلقائياً مع كل نسخة محددة.",
    "export.cancel": "إلغاء",
    "export.confirm": "تصدير المحدد",
    "export.model": "نسخة {n}",
    "export.select_all": "تحديد الكل",
    "export.clear_all": "إلغاء تحديد الكل",
    "common.close": "إغلاق",
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

export function localizedError(detail, fallbackKey, params) {
  const message = String(detail || "").trim();
  if (current === "ar") {
    return /[\u0600-\u06ff]/.test(message) ? message : t(fallbackKey, params);
  }
  return message || t(fallbackKey, params);
}

export function stageLabel(stage) {
  const key = `stage.${String(stage || "").toLowerCase()}`;
  const translated = t(key);
  return translated === key ? t("stage.queued") : translated;
}

export function examStageLabel(stage) {
  const normalized = String(stage || "queued").toLowerCase();
  const key = `exam_stage.${normalized}`;
  const translated = t(key);
  return translated === key ? t("exam_stage.queued") : translated;
}

export function directionForText(value, fallback = "en") {
  const text = String(value || "");
  const arabic = (text.match(/[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]/g) || []).length;
  const latin = (text.match(/[A-Za-z\u00c0-\u024f]/g) || []).length;
  if (!arabic && !latin) return fallback === "ar" ? "rtl" : "ltr";
  return arabic * 10 >= latin ? "rtl" : "ltr";
}

export function applyContentDirection(element, value, fallback = "en") {
  if (!element) return element;
  const direction = directionForText(value, fallback);
  element.dir = direction;
  element.lang = direction === "rtl" ? "ar" : "en";
  return element;
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
  for (const el of document.querySelectorAll("[data-i18n-aria]")) {
    el.setAttribute("aria-label", t(el.dataset.i18nAria));
  }
  const switcher = document.getElementById("langSwitch");
  if (switcher) switcher.textContent = rtl ? "EN" : "ع";
  document.dispatchEvent(new CustomEvent("draftwork:language-changed", { detail: { lang } }));
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
