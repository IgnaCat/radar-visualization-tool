import { useRef, useCallback } from "react";
import { Paper } from "@mui/material";
import { useDraggableResizable } from "../../hooks/useDraggableResizable";

/**
 * Componente Paper que soporta drag y resize para diálogos MUI.
 *
 * Props:
 * - dialogStateRef: ref con { position, size } para persistir estado entre renders
 * - defaultWidth, defaultHeight: dimensiones iniciales
 * - minWidth, minHeight: dimensiones mínimas permitidas
 * - ...props: el resto se pasa directamente al Paper de MUI
 */
export default function DraggableDialogPaper({
  dialogStateRef,
  defaultWidth = 600,
  defaultHeight = 500,
  minWidth = 400,
  minHeight = 300,
  ...props
}) {
  const savedPos = dialogStateRef.current.position;
  const savedSize = dialogStateRef.current.size;
  const initWidth = savedSize?.width || defaultWidth;
  const initHeight = savedSize?.height || defaultHeight;
  const initX = savedPos?.x ?? 0;
  const initY = savedPos?.y ?? 0;

  const { nodeRef, position, size, cursor, handleMouseDown, handleMouseMove } =
    useDraggableResizable({
      initialX: initX,
      initialY: initY,
      initialWidth: initWidth,
      initialHeight: initHeight,
      minWidth,
      minHeight,
      edgeSize: 15,
      centerOnMount: !savedPos,
      onPositionChange: (pos) => {
        dialogStateRef.current.position = pos;
      },
      onSizeChange: (sz) => {
        dialogStateRef.current.size = sz;
      },
    });

  return (
    <Paper
      {...props}
      ref={nodeRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      sx={{
        // "&&" duplica el selector CSS para ganar especificidad sobre
        // los estilos internos de MUI Dialog (.MuiDialog-paper) que
        // aplican position:relative y margin:32px.
        "&&": {
          position: "fixed",
          left: position.x,
          top: position.y,
          width: size.width,
          height: size.height,
          m: 0,
          maxWidth: "none",
          maxHeight: "none",
        },
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        pointerEvents: "auto",
        cursor: cursor,
        zIndex: 1300,
        userSelect: "none",
      }}
    />
  );
}

/**
 * Hook que encapsula el patrón dialogStateRef + PaperComponent para diálogos
 * con drag y resize.
 *
 * Uso:
 *   const { PaperComponent } = useDraggableDialogPaper({ defaultWidth: 600, defaultHeight: 500 });
 *   <Dialog ... PaperComponent={PaperComponent} />
 *
 * @param {object} options
 * @param {number} options.defaultWidth
 * @param {number} options.defaultHeight
 * @param {number} options.minWidth
 * @param {number} options.minHeight
 * @returns {{ PaperComponent: React.ComponentType, dialogStateRef: React.MutableRefObject }}
 */
export function useDraggableDialogPaper({
  defaultWidth = 600,
  defaultHeight = 500,
  minWidth = 400,
  minHeight = 300,
} = {}) {
  const dialogStateRef = useRef({ position: null, size: null });

  const PaperComponent = useCallback(
    (props) => (
      <DraggableDialogPaper
        {...props}
        dialogStateRef={dialogStateRef}
        defaultWidth={defaultWidth}
        defaultHeight={defaultHeight}
        minWidth={minWidth}
        minHeight={minHeight}
      />
    ),
    // Las dimensiones son estáticas por diseño; dialogStateRef es un ref y no cambia.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return { PaperComponent, dialogStateRef };
}
