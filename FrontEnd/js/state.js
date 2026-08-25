// Shared application state (replaces former inline-script globals).
export const state = {
  counts: { mcq: 10, tf: 5, fitb: 5, why: 3, essay: 2 },
  enabled: { mcq: true, tf: true, fitb: true, why: true, essay: true },
  numModels: 1,
  difficulty: "easy",
  uploadDone: false,
  currentDocId: null,
  selectedChildren: new Set(),
  examId: null,
  exams: [],
  rawExam: "",
  teacherEmail: null,
};
