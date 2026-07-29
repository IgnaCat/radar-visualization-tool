import { CircleMarker, Tooltip } from "react-leaflet";
import "./LocateButton.css";

/**
 * Renders a Google Maps-style pulsating blue dot at the user's location.
 * Clicking the marker dismisses it.
 *
 * @param {[number, number] | null} position  - [lat, lon] or null to hide
 * @param {() => void}             onDismiss  - called when user clicks the marker
 */
export default function LocateButton({ position, onDismiss }) {
  if (!position) return null;

  return (
    <>
      {/* Outer pulsating ring */}
      <CircleMarker
        center={position}
        radius={14}
        pathOptions={{
          fillColor: "#1a73e8",
          fillOpacity: 0.15,
          color: "#1a73e8",
          weight: 1,
          opacity: 0.4,
        }}
        className="user-location-pulse"
        eventHandlers={{ click: onDismiss }}
      />

      {/* Inner filled dot */}
      <CircleMarker
        center={position}
        radius={6}
        pathOptions={{
          fillColor: "#1a73e8",
          fillOpacity: 1,
          color: "white",
          weight: 2,
        }}
        eventHandlers={{ click: onDismiss }}
      >
        <Tooltip direction="top" offset={[0, -8]}>
          Tu ubicación · click para quitar
        </Tooltip>
      </CircleMarker>
    </>
  );
}
