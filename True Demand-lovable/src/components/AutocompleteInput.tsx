import { useEffect, useMemo, useRef, useState } from "react";
import type { Suggestion } from "@/lib/types";
import { api } from "@/lib/api";
import { TypeBadge } from "./TypeBadge";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

function getCurrentToken(text: string, caret: number) {
  const before = text.slice(0, caret);
  const m = before.match(/([A-Za-z0-9_-]*)$/);
  const token = m ? m[1] : "";
  const start = caret - token.length;
  return { token, start };
}

export function AutocompleteInput({ value, onChange, onSubmit, disabled }: Props) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<Suggestion[]>([]);
  const [active, setActive] = useState(0);
  const [popPos, setPopPos] = useState<{ left: number; top: number } | null>(null);
  const [caret, setCaret] = useState(0);

  const token = useMemo(() => getCurrentToken(value, caret).token, [value, caret]);

  useEffect(() => {
    if (!token || token.length < 1) {
      setOpen(false);
      return;
    }
    let cancelled = false;
    api.autocomplete(token, value.slice(0, caret)).then((res) => {
      if (cancelled) return;
      setItems(res.slice(0, 8));
      setActive(0);
      setOpen(res.length > 0);
    });
    return () => {
      cancelled = true;
    };
  }, [token, value, caret]);

  // position the popover beneath the caret using a mirror div
  useEffect(() => {
    if (!open) return;
    const ta = taRef.current;
    if (!ta) return;
    const mirror = document.createElement("div");
    const style = window.getComputedStyle(ta);
    for (const prop of [
      "boxSizing",
      "width",
      "height",
      "padding",
      "border",
      "fontFamily",
      "fontSize",
      "fontWeight",
      "lineHeight",
      "letterSpacing",
      "whiteSpace",
      "wordWrap",
      "overflowWrap",
    ]) {
      // @ts-expect-error indexing
      mirror.style[prop] = style[prop];
    }
    mirror.style.position = "absolute";
    mirror.style.visibility = "hidden";
    mirror.style.whiteSpace = "pre-wrap";
    mirror.style.top = "0";
    mirror.style.left = "0";
    const before = value.slice(0, caret);
    mirror.textContent = before;
    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    mirror.appendChild(marker);
    document.body.appendChild(mirror);
    const rect = ta.getBoundingClientRect();
    const markerRect = marker.getBoundingClientRect();
    const mirrorRect = mirror.getBoundingClientRect();
    const left = rect.left + (markerRect.left - mirrorRect.left) - ta.scrollLeft;
    const top = rect.top + (markerRect.top - mirrorRect.top) - ta.scrollTop + markerRect.height + 4;
    setPopPos({ left, top });
    document.body.removeChild(mirror);
  }, [open, value, caret, items.length]);

  function insertSuggestion(s: Suggestion) {
    const { start } = getCurrentToken(value, caret);
    const next = value.slice(0, start) + s.label + value.slice(caret);
    const newCaret = start + s.label.length;
    onChange(next);
    setOpen(false);
    queueMicrotask(() => {
      const ta = taRef.current;
      if (ta) {
        ta.focus();
        ta.setSelectionRange(newCaret, newCaret);
        setCaret(newCaret);
      }
    });
  }

  return (
    <div className="relative">
      <textarea
        ref={taRef}
        rows={2}
        value={value}
        disabled={disabled}
        placeholder="Ask about metrics, dimensions, entities… (e.g. 'true demand for AURIX TC4x in EMEA')"
        className="w-full resize-none rounded-md border border-input bg-card px-3.5 py-2.5 text-[14px] leading-[1.5] outline-none focus:border-ring focus:ring-2 focus:ring-ring/20 transition-colors"
        onChange={(e) => {
          onChange(e.target.value);
          setCaret(e.target.selectionStart);
        }}
        onClick={(e) => setCaret(e.currentTarget.selectionStart)}
        onKeyUp={(e) => setCaret(e.currentTarget.selectionStart)}
        onKeyDown={(e) => {
          if (open && items.length > 0) {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => (a + 1) % items.length);
              return;
            }
            if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => (a - 1 + items.length) % items.length);
              return;
            }
            if (e.key === "Enter" || e.key === "Tab") {
              e.preventDefault();
              insertSuggestion(items[active]);
              return;
            }
            if (e.key === "Escape") {
              setOpen(false);
              return;
            }
          }
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />
      {open && popPos && items.length > 0 && (
        <div
          className="fixed z-50 w-80 rounded-md border border-border bg-popover shadow-lg overflow-hidden"
          style={{ left: popPos.left, top: popPos.top }}
        >
          <ul className="max-h-72 overflow-auto py-1">
            {items.map((s, i) => (
              <li
                key={s.id}
                onMouseDown={(e) => {
                  e.preventDefault();
                  insertSuggestion(s);
                }}
                onMouseEnter={() => setActive(i)}
                className="px-3 py-1.5 cursor-pointer flex items-start gap-2"
                style={{ backgroundColor: i === active ? "var(--accent)" : "transparent" }}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium truncate">{s.label}</span>
                    <TypeBadge type={s.type} />
                  </div>
                  {s.description && (
                    <div className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {s.description}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
          <div className="px-3 py-1.5 border-t border-border bg-muted/50 text-[10px] text-muted-foreground flex items-center justify-between">
            <span>↑↓ navigate · Tab/Enter insert · Esc close</span>
            <span>⌘+Enter to ask</span>
          </div>
        </div>
      )}
    </div>
  );
}
