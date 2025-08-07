export default function ColorLegend() {
  const legend = [
    { value: 70, color: "purple", label: "Lluvia muy intensa y granizo" },
    { value: 60, color: "red", label: "Lluvia muy intensa y granizo" },
    { value: 50, color: "orange", label: "Lluvia intensa" },
    { value: 40, color: "yellow", label: "Lluvia intensa" },
    { value: 30, color: "green", label: "Lluvia moderada" },
    { value: 20, color: "blue", label: "Lluvia leve" },
    { value: 10, color: "lightblue", label: "Llovisna" },
    { value: 0, color: "gray", label: "Neblina" },
    { value: -10, color: "white", label: "Nubes no precipitantes" },
  ];

  return (
    <div
      style={{
        position: "absolute",
        right: 10,
        top: 100,
        zIndex: 1000,
        background: "rgba(255,255,255,0.8)",
        padding: "10px",
        borderRadius: "8px",
      }}
    >
      {legend.map((item) => (
        <div key={item.value} style={{ color: item.color }}>
          {item.value}: {item.label || ""}
        </div>
      ))}
    </div>
  );
}
