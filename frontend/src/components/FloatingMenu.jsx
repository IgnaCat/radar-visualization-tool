import { SpeedDial, SpeedDialAction } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import VisibilityIcon from "@mui/icons-material/Visibility";

export default function FloatingMenu({ onUploadClick, onChangeProductClick }) {
  const actions = [
    { icon: <CloudUploadIcon />, name: "Subir archivo", action: onUploadClick },
    {
      icon: <VisibilityIcon />,
      name: "Cambiar producto",
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
