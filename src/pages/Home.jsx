import styles from './Home.module.css';
import Lottie from 'lottie-react';
import animacion from '../assets/animacion.json';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className={styles.homeContainer}>
      <div className={styles.textSection}>
        <h2>🎓 Bienvenido a la Plataforma de Predicción de Rendimiento Académico</h2>
        <p>
          Esta plataforma utiliza técnicas de Inteligencia Artificial para predecir tu ponderado futuro
          en base a información académica, hábitos de estudio, situación familiar y mucho más.
        </p>
        <p>
          🔐 Tus datos son completamente confidenciales y utilizados exclusivamente para fines de análisis académico.
        </p>
        <button
          onClick={() => navigate('/prediccion')}
          className={styles.startButton}
        >
          ¡Quiero saber mi próximo ponderado!
        </button>
      </div>

      <div className={styles.animationSection}>
        <Lottie animationData={animacion} loop autoplay style={{ height: 300 }} />
      </div>
    </div>
  );
};

export default Home;
