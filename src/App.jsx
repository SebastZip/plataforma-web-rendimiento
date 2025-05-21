import { useState } from 'react'
import LoginPage from './pages/LoginPage.jsx'
import PrediccionForm from './pages/PrediccionForm.jsx'

function App() {
  const [userSession, setUserSession] = useState(null)

  return (
    <>
      {userSession
        ? <PrediccionForm usuario={userSession} />
        : <LoginPage onLogin={setUserSession} />}
    </>
  )
}

export default App
