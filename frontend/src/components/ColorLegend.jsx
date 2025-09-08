export default function ColorLegend() {
  const legend = [
    {
      value: "?",
      color: "grey",
      label: "COLMAX(Z)",
      letter_color: "black",
    },
    {
      value: 70,
      color: "#FF29E3",
      label: "Lluvia muy intensa y granizo",
      letter_color: "white",
    },
    {
      value: 60,
      color: "#FF2A98",
      label: "Lluvia muy intensa y granizo",
      letter_color: "white",
    },
    {
      value: 50,
      color: "#FF2A0C",
      label: "Lluvia intensa",
      letter_color: "white",
    },
    {
      value: 40,
      color: "#f7a600",
      label: "Lluvia intensa",
      letter_color: "black",
    },
    {
      value: 30,
      color: "#EAF328",
      label: "Lluvia moderada",
      letter_color: "black",
    },
    {
      value: 20,
      color: "#00AD5A",
      label: "Lluvia leve",
      letter_color: "black",
    },
    { value: 10, color: "#00E68A", label: "Llovizna", letter_color: "black" },
    { value: 0, color: "#95b4dc", label: "Neblina", letter_color: "black" },
    {
      value: -10,
      color: "#f0f6f2",
      label: "Nubes no precipitantes",
      letter_color: "black",
    },
  ];

  return (
    <div
      style={{
        position: "absolute",
        left: 40,
        bottom: 10,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        gap: "5px",
      }}
    >
      {legend.map((item) => (
        <div
          key={item.value}
          title={`${item.label}`}
          style={{
            width: "26px",
            height: "26px",
            borderRadius: "50%",
            backgroundColor: item.color,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: "bold",
            color: item.letter_color,
            fontSize: "16px",
          }}
        >
          {item.value || item.value === 0 ? item.value : "?"}
        </div>
      ))}
    </div>
  );
}
