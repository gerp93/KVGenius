/** The Generate page's prompt tabs as the sidebar shows them. Generate owns the tabs (each is a full copy of its
 * form) and publishes this; the sidebar only lists them and calls back. */
export interface PromptTabItem {
  id: string;
  /** What the tab is called: its own name, else the start of its prompt. */
  label: string;
  mode: 'image' | 'video';
}

export interface PromptTabsModel {
  tabs: PromptTabItem[];
  activeId: string;
  canAdd: boolean;
  maxTabs: number;
  select: (id: string) => void;
  add: () => void;
  close: (id: string) => void;
  rename: (id: string, name: string) => void;
}
