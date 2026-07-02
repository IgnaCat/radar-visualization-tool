import {
  Backdrop,
  CircularProgress,
  Box,
  LinearProgress,
  Typography,
} from "@mui/material";

export default function Loader({ open, uploadProgress = null }) {
  const showProgress = uploadProgress !== null && uploadProgress >= 0;
  const percent = Math.round((uploadProgress ?? 0) * 100);
  const uploading = showProgress && percent < 100;

  return (
    <Backdrop
      sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
      open={open}
    >
      {showProgress ? (
        <Box sx={{ width: 320, textAlign: "center" }}>
          {uploading ? (
            <LinearProgress
              variant="determinate"
              value={percent}
              sx={{
                height: 8,
                borderRadius: 4,
                mb: 1,
                backgroundColor: "rgba(255,255,255,0.2)",
                "& .MuiLinearProgress-bar": {
                  borderRadius: 4,
                },
              }}
            />
          ) : (
            <CircularProgress color="inherit" size={40} sx={{ mb: 2 }} />
          )}
          <Typography variant="body2" color="inherit">
            {uploading
              ? `Subiendo archivos… ${percent}%`
              : "Procesando archivos…"}
          </Typography>
        </Box>
      ) : (
        <CircularProgress color="inherit" />
      )}
    </Backdrop>
  );
}
