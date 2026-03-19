import { useRef, useState, useCallback, useEffect } from "react";

export function useDraggableElement(
  defaultX = 0,
  defaultY = 0,
  fromBottom = false,
) {
  const elementRef = useRef(null);
  const [position, setPosition] = useState({ x: defaultX, y: defaultY });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const hasUserMovedRef = useRef(false);

  const handleMouseDown = useCallback(
    (e) => {
      if (!elementRef.current) return;

      // Iniciar drag desde cualquier click dentro del elemento
      setIsDragging(true);
      const rect = elementRef.current.getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: fromBottom ? rect.bottom - e.clientY : e.clientY - rect.top,
      });
    },
    [fromBottom],
  );

  const handleMouseMove = useCallback(
    (e) => {
      if (!isDragging || !elementRef.current) return;

      const newX = e.clientX - dragOffset.x;
      const newY = fromBottom
        ? window.innerHeight - e.clientY - dragOffset.y
        : e.clientY - dragOffset.y;

      setPosition({ x: newX, y: newY });
      hasUserMovedRef.current = true;
    },
    [isDragging, dragOffset, fromBottom],
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Attach listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Keep initial position in sync when defaults change (e.g. legend slot assignment),
  // but never override a position manually moved by the user.
  useEffect(() => {
    if (!hasUserMovedRef.current) {
      setPosition({ x: defaultX, y: defaultY });
    }
  }, [defaultX, defaultY]);

  return {
    elementRef,
    position,
    setPosition,
    style: fromBottom
      ? {
          position: "fixed",
          left: `${position.x}px`,
          bottom: `${position.y}px`,
          touchAction: "none",
          zIndex: isDragging ? 10001 : 1000,
          cursor: "move",
        }
      : {
          position: "fixed",
          left: `${position.x}px`,
          top: `${position.y}px`,
          touchAction: "none",
          zIndex: isDragging ? 10001 : 1000,
          cursor: "move",
        },
    onMouseDown: handleMouseDown,
  };
}
