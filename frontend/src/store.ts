import { create } from "zustand";
import type { AnalysisResult } from "./types";

interface State {
  result: AnalysisResult | null;
  open: boolean;
  tab: number;
  show: (r: AnalysisResult) => void;
  close: () => void;
  setTab: (t: number) => void;
}

// Cerrar el panel sólo oculta; el resultado y la pestaña se conservan.
export const useStore = create<State>((set) => ({
  result: null,
  open: false,
  tab: 0,
  show: (result) => set({ result, open: true, tab: 0 }),
  close: () => set({ open: false }),
  setTab: (tab) => set({ tab }),
}));
