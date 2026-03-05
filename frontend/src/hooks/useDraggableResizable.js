import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Hook personalizado para hacer un componente draggable y resizable
 * @param {object} options - Opciones de configuración
 * @param {number} options.initialX - Posición X inicial
 * @param {number} options.initialY - Posición Y inicial
 * @param {number} options.initialWidth - Ancho inicial
 * @param {number} options.initialHeight - Alto inicial
 * @param {number} options.minWidth - Ancho mínimo (default: 400)
 * @param {number} options.minHeight - Alto mínimo (default: 300)
 * @param {number} options.edgeSize - Tamaño del área de resize en px (default: 12)
 */
export function useDraggableResizable({
    initialX,
    initialY,
    initialWidth,
    initialHeight,
    minWidth = 400,
    minHeight = 300,
    edgeSize = 12,
    onPositionChange,
    onSizeChange,
} = {}) {
    const nodeRef = useRef(null);
    const [position, setPositionState] = useState({ x: initialX || 100, y: initialY || 100 });
    const [size, setSizeState] = useState({
        width: initialWidth || 600,
        height: initialHeight || 600
    });
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);
    const [resizeDir, setResizeDir] = useState(null);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [cursor, setCursor] = useState('default');

    // Wrappers para notificar cambios
    const setPosition = useCallback((newPos) => {
        setPositionState(newPos);
        if (onPositionChange) {
            onPositionChange(newPos);
        }
    }, [onPositionChange]);

    const setSize = useCallback((newSize) => {
        setSizeState(newSize);
        if (onSizeChange) {
            onSizeChange(newSize);
        }
    }, [onSizeChange]);

    // Detectar en qué dirección debe hacerse el resize basándose en la posición del mouse
    const getResizeDirection = useCallback((e) => {
        if (!nodeRef.current) return null;
        const rect = nodeRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const onRight = x >= rect.width - edgeSize;
        const onBottom = y >= rect.height - edgeSize;
        const onLeft = x <= edgeSize;
        const onTop = y <= edgeSize;

        // Priorizar esquinas
        if (onRight && onBottom) return 'se';
        if (onLeft && onBottom) return 'sw';
        if (onRight && onTop) return 'ne';
        if (onLeft && onTop) return 'nw';

        // Luego bordes
        if (onRight) return 'e';
        if (onBottom) return 's';
        if (onLeft) return 'w';
        if (onTop) return 'n';

        return null;
    }, [edgeSize]);

    // Obtener el cursor apropiado para cada dirección
    const getCursorForDirection = useCallback((dir) => {
        const cursors = {
            'n': 'ns-resize',
            's': 'ns-resize',
            'e': 'ew-resize',
            'w': 'ew-resize',
            'ne': 'nesw-resize',
            'sw': 'nesw-resize',
            'nw': 'nwse-resize',
            'se': 'nwse-resize',
        };
        return cursors[dir] || 'default';
    }, []);

    // Manejar inicio de drag o resize
    const handleMouseDown = useCallback((e) => {
        const dir = getResizeDirection(e);

        if (dir) {
            // Iniciar resize
            setIsResizing(true);
            setResizeDir(dir);
            setDragStart({ x: e.clientX, y: e.clientY });
            e.preventDefault();
            e.stopPropagation();
            return;
        }

        // Ignorar clicks en elementos interactivos para el drag
        if (
            e.target.closest('button') ||
            e.target.closest('input') ||
            e.target.closest('textarea') ||
            e.target.closest('.MuiSlider-root') ||
            e.target.closest('select') ||
            e.target.closest('.MuiAutocomplete-root') ||
            e.target.tagName === 'IMG' ||
            e.target.closest('svg') ||
            e.target.closest('a')
        ) {
            return;
        }

        // Iniciar drag
        setIsDragging(true);
        setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }, [getResizeDirection, position]);

    // Manejar movimiento del mouse
    const handleMouseMove = useCallback((e) => {
        if (!nodeRef.current) return;

        if (isResizing) {
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            const newSize = { ...size };
            const newPos = { ...position };

            // Ajustar según la dirección del resize
            if (resizeDir.includes('e')) {
                newSize.width = Math.max(minWidth, size.width + dx);
            }
            if (resizeDir.includes('s')) {
                newSize.height = Math.max(minHeight, size.height + dy);
            }
            if (resizeDir.includes('w')) {
                const newWidth = Math.max(minWidth, size.width - dx);
                if (newWidth > minWidth) {
                    newPos.x = position.x + dx;
                    newSize.width = newWidth;
                }
            }
            if (resizeDir.includes('n')) {
                const newHeight = Math.max(minHeight, size.height - dy);
                if (newHeight > minHeight) {
                    newPos.y = position.y + dy;
                    newSize.height = newHeight;
                }
            }

            setSize(newSize);
            setPosition(newPos);
            setDragStart({ x: e.clientX, y: e.clientY });
        } else if (isDragging) {
            // Mover el diálogo
            setPosition({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y,
            });
        } else {
            // Actualizar cursor cuando no está arrastrando/redimensionando
            const dir = getResizeDirection(e);
            setCursor(dir ? getCursorForDirection(dir) : 'move');
        }
    }, [
        isResizing,
        isDragging,
        dragStart,
        position,
        size,
        resizeDir,
        minWidth,
        minHeight,
        getResizeDirection,
        getCursorForDirection,
        setPosition,
        setSize
    ]);

    // Manejar fin de drag o resize
    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
        setIsResizing(false);
        setResizeDir(null);
    }, []);

    // Agregar/remover event listeners globales
    useEffect(() => {
        if (isDragging || isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);

            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            };
        }
    }, [isDragging, isResizing, handleMouseMove, handleMouseUp]);

    // Determinar el cursor final
    const finalCursor = isResizing
        ? getCursorForDirection(resizeDir)
        : (isDragging ? 'move' : cursor);

    return {
        nodeRef,
        position,
        size,
        cursor: finalCursor,
        handleMouseDown,
        handleMouseMove,
        isDragging,
        isResizing,
    };
}
