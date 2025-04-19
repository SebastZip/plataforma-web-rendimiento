function FormularioRiesgo() {
    return (
      <div className="min-h-screen bg-[#f1f6ff] flex items-center justify-center px-4 py-8">
        <div className="bg-white shadow-xl rounded-2xl w-full max-w-2xl p-8">
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Evaluación de Riesgo Académico</h2>
  
          <form className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Código del estudiante</label>
              <input
                type="text"
                placeholder="20230001FISI"
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-300"
              />
            </div>
  
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Ciclo actual</label>
              <select className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-300">
                <option value="">Selecciona...</option>
                <option value="2025-I">2025-I</option>
                <option value="2025-II">2025-II</option>
              </select>
            </div>
  
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Estado emocional general</label>
              <textarea
                placeholder="Describe cómo te sientes actualmente..."
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-blue-300"
                rows={3}
              ></textarea>
            </div>
  
            <div className="flex justify-end">
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md font-semibold transition"
              >
                Evaluar riesgo
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }
  
  export default FormularioRiesgo;
  