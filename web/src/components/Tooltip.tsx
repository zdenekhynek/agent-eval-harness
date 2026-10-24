import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

// One floating tooltip for the whole page. Content is React nodes, so agent
// names and other data are always rendered as text, never as HTML.
interface Tip {
  content: ReactNode;
  x: number;
  y: number;
}

interface TipApi {
  show: (content: ReactNode, target: { clientX: number; clientY: number } | Element) => void;
  hide: () => void;
}

const TipContext = createContext<TipApi>({ show: () => {}, hide: () => {} });

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const show = useCallback<TipApi["show"]>((content, target) => {
    if (target instanceof Element) {
      const r = target.getBoundingClientRect();
      setTip({ content, x: r.left + r.width / 2, y: r.top });
    } else {
      setTip({ content, x: target.clientX, y: target.clientY });
    }
  }, []);
  const hide = useCallback(() => setTip(null), []);

  let style: React.CSSProperties | undefined;
  if (tip) {
    const flip = tip.x > window.innerWidth - 300;
    style = {
      top: Math.max(8, tip.y - 12),
      left: flip ? undefined : tip.x + 14,
      right: flip ? window.innerWidth - tip.x + 14 : undefined,
      transform: "translateY(-100%)",
    };
  }
  return (
    <TipContext.Provider value={{ show, hide }}>
      {children}
      {tip && (
        <div className="tooltip" role="status" style={style}>
          {tip.content}
        </div>
      )}
    </TipContext.Provider>
  );
}

export const useTooltip = () => useContext(TipContext);

/** Hover and keyboard-focus handlers that show `content` for one mark. */
export function useTipHandlers() {
  const { show, hide } = useTooltip();
  return (content: ReactNode) => ({
    onPointerMove: (e: React.PointerEvent) => show(content, e),
    onPointerLeave: hide,
    onFocus: (e: React.FocusEvent) => show(content, e.currentTarget),
    onBlur: hide,
  });
}
