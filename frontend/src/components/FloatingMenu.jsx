import { SpeedDial, SpeedDialAction } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import VisibilityIcon from "@mui/icons-material/Visibility";
import PercentIcon from "@mui/icons-material/Percent";
export default function FloatingMenu({
  onUploadClick,
  onChangeProductClick,
  onPseudoRhiClick,
  onAreaStatsClick,
}) {
  const actions = [
    { icon: <CloudUploadIcon />, name: "Subir archivo", action: onUploadClick },
    {
      icon: <VisibilityIcon />,
      name: "Opciones de visualización",
      action: onChangeProductClick,
    },
    {
      icon: <MenuIcon />,
      name: "Generar Pseudo-RHI",
      action: onPseudoRhiClick,
    },
    {
      icon: <PercentIcon />,
      name: "Estadísticas",
      action: onAreaStatsClick,
    },
  ];

  return (
    <SpeedDial
      ariaLabel="Menu"
      sx={{ position: "absolute", bottom: 16, right: 16 }}
      icon={<MenuIcon />}
    >
      {actions.map((act) => (
        <SpeedDialAction
          key={act.name}
          icon={act.icon}
          tooltipTitle={act.name}
          onClick={act.action}
        />
      ))}
    </SpeedDial>
  );
}
