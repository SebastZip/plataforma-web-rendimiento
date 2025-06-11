import { Link, Outlet } from 'react-router-dom';
import styles from './SidebarLayout.module.css';

const SidebarLayout = ({ usuario, onLogout }) => {
  return (
    <div className={styles.dashboard}>
      <aside className={styles.sidebar}>
        <h2>Menú</h2>
        <ul>
          <li><Link to="/">🏠 Inicio</Link></li>
          <li><Link to="/prediccion">📈 Predecir</Link></li>
          <li><Link to="/historial">📊 Historial</Link></li>
        </ul>
      </aside>

      <main className={styles.mainContent}>
        <div className={styles.topBar}>
          <div className={styles.saludo}>
            <span>¡Buenos días, <strong>{usuario.nombres}</strong>!</span>
          </div>

          <div className={styles.tituloCentrado}>
            <span className={styles.glowText}>Plataforma de Predicción de Rendimiento Académico</span>
          </div>

          <div className={styles.cerrarSesion}>
            <button className={styles.logoutButton} onClick={onLogout}>Cerrar Sesión</button>
          </div>
        </div>

        <Outlet />
      </main>
    </div>
  );
};

export default SidebarLayout;
