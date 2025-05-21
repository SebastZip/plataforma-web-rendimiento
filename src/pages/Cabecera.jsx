import styles from './Cabecera.module.css';

const Cabecera = ({ nombre, onLogout }) => {
  const obtenerSaludo = () => {
    const hora = new Date().getHours();
    if (hora < 12) return '¡Buenos días!';
    if (hora < 18) return '¡Buenas tardes!';
    return '¡Buenas noches!';
  };

  return (
    <div className={styles.headerBar}>
      <div className={styles.left}>
        {obtenerSaludo()}
        {nombre && <span style={{ marginLeft: '8px', fontWeight: 'bold' }}>{nombre}</span>}
      </div>
      <h1 className={`${styles.center} ${styles.glowText}`}>
        Plataforma de Predicción de Rendimiento Académico
      </h1>
      <div className={styles.right}>
        <button className={styles.logoutButton} onClick={onLogout}>
          Cerrar Sesión
        </button>
      </div>
    </div>
  );
};

export default Cabecera;
