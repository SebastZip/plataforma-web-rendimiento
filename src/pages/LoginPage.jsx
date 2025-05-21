import { useState, useRef, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import styles from './LoginPage.module.css'

import fondo from '../assets/fondo.jpg'
import iconUser from '../assets/login/v98_31.png'
import iconPass from '../assets/login/v98_34.png'
import logoUNMSM from '../assets/login/v98_36.png'

const LoginPage = ({ onLogin }) => {
  const [codigo, setCodigo] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const backgroundRef = useRef(null)

  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 10
      const y = (e.clientY / window.innerHeight - 0.5) * 10
      if (backgroundRef.current) {
        backgroundRef.current.style.transform = `translate(${x}px, ${y}px) scale(1.05)`
      }
    }

    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  const handleLogin = async (e) => {
    e.preventDefault()
    setError('')

    const { data, error: fetchError } = await supabase
      .from('users')
      .select('*')
      .eq('codigo', codigo)
      .single()

    if (fetchError || !data) {
      setError('❌ Código no encontrado')
      return
    }

    if (data.password !== password) {
      setError('❌ Contraseña incorrecta')
      return
    }

    onLogin(data)
  }

  return (
    <div className={styles.loginContainer}>
      <img
        ref={backgroundRef}
        src={fondo}
        alt="Fondo desenfocado"
        className={styles.backgroundBlur}
      />

      <div className={styles.welcomeText}>
        <h1>¡Bienvenido!</h1>
      </div>

      <div className={styles.loginBox}>
        <img src={logoUNMSM} alt="Logo UNMSM" className={styles.logoUnmsm} />

        <form onSubmit={handleLogin} className={styles.formGrid}>
          <div className={styles.inputGroup}>
            <div className={styles.iconWrapper}>
              <img src={iconUser} alt="icono usuario" className={styles.inputIcon} />
            </div>
            <input
              type="text"
              placeholder="Código de usuario"
              value={codigo}
              onChange={(e) => setCodigo(e.target.value)}
              required
            />
          </div>
          <div className={styles.inputGroup}>
            <div className={styles.iconWrapper}>
              <img src={iconPass} alt="icono contraseña" className={styles.inputIcon} />
            </div>
            <input
              type="password"
              placeholder="Contraseña"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className={styles.submitButton}>Ingresar</button>
          {error && <p className={styles.errorMessage}>{error}</p>}
        </form>
      </div>
    </div>
  )
}

export default LoginPage
