import { Link as RouterLink, Route, Routes, useLocation } from "react-router-dom";
import { AppBar, Button, Container, Toolbar, Typography } from "@mui/material";
import UploadComponent from "./components/UploadComponent";
import ResultSidebarComponent from "./components/ResultSidebarComponent";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  const { pathname } = useLocation();
  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>Validación Documental · POC</Typography>
          <Button color="inherit" component={RouterLink} to="/" disabled={pathname === "/"}>Analizar</Button>
          <Button color="inherit" component={RouterLink} to="/history" disabled={pathname === "/history"}>Historial</Button>
        </Toolbar>
      </AppBar>
      <Container sx={{ py: 4 }} maxWidth="md">
        <Routes>
          <Route path="/" element={<UploadComponent />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Container>
      <ResultSidebarComponent />
    </>
  );
}
