import { SpeedDial, SpeedDialAction } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SettingsIcon from "@mui/icons-material/Settings";
export default function FloatingMenu({ onUploadClick, onChangeProductClick }) {
  const actions = [
    { icon: <CloudUploadIcon />, name: "Subir archivo", action: onUploadClick },
    {
      icon: <SettingsIcon />,
      name: "Configuración",
      action: onChangeProductClick,
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
