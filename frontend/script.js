(function(){
  const form = document.getElementById('recomendar-form');
  const feedbackEl = document.getElementById('feedback');
  const rankingBody = document.getElementById('ranking-body');
  const submitBtn = document.getElementById('submit-btn');
  const segmentoSelect = document.getElementById('segmento');
  const canalSelect = document.getElementById('canal');
  const provinciaSelect = document.getElementById('provincia');

  if(!form) return;

  // Listas conforme schemas.py
  const SEGMENTOS = ["top","particulares","universitario"];
  const CANAL_LIST = ['KAT','KFC','KFA','007','013','KCG','KAF','KAW','KCH','KGX','RED','KAS','KHN','KAY','KBW','KAG','KBZ','KHE','Desconhecido','KAH','KAA','KAU','KHK','KHL','KHM','KAZ','KAR','KCF','KEG','KEY','KCN','KCC','KBH','KDX','KBF','KBD','KCB','KBU','KES','KBO','KCI','KAB','KCU','KAC','KDR','KDC','KEF','KCM','KDY','KCA','KDS','KDU','KEA','KAE','KCL','KAD','KCD','KFD','KDO','KBQ','KFP','KEI','KHC','KHO','KEW','KDM','KEZ','KEL','KAP','KAI','KAJ','KEJ','KBJ','KED','KAO','KAK','KAL','KBR','KEN','KBB','KBS','K00','KAQ','KFK','KAM','KHQ','KBM','KGY','KFU','KFM','KFN','KFH','KFG','KFT','KFJ','KFS','KGV','KFF','KBG','KHD','KEH','KHF','KHP'];
  const PROVINCIAS = ['MADRID','GRANADA','MALAGA','BARCELONA','ALICANTE','ALMERIA','VALLADOLID','SEVILLA','ZAMORA','GIRONA','VALENCIA','HUELVA','GIPUZKOA','ASTURIAS','BALEARS, ILLES','CANTABRIA','JAEN','SANTA CRUZ DE TENERIFE','MURCIA','LERIDA','CUENCA','CIUDAD REAL','BIZKAIA','CADIZ','ALBACETE','TARRAGONA','CORUÑA, A','BURGOS','BADAJOZ','ALAVA','PALMAS, LAS','RIOJA, LA','MELILLA','OURENSE','ZARAGOZA','NAVARRA','GUADALAJARA','CASTELLON','PONTEVEDRA','SALAMANCA','CEUTA','TOLEDO','CORDOBA','HUESCA','SORIA','CACERES','LUGO','LEON','PALENCIA','AVILA','TERUEL','SEGOVIA'];

  function toTitleCaseSegmento(s){
    if(s === 'top') return 'Top';
    if(s === 'particulares') return 'Particulares';
    if(s === 'universitario') return 'Universitário';
    return s;
  }

  function populateSelect(selectEl, values, labelFn){
    if(!selectEl) return;
    selectEl.innerHTML = '';
    values.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = labelFn ? labelFn(v) : v;
      selectEl.appendChild(opt);
    });
  }

  // Popular selects de acordo com schemas
  populateSelect(segmentoSelect, SEGMENTOS, toTitleCaseSegmento);
  populateSelect(canalSelect, CANAL_LIST);
  populateSelect(provinciaSelect, PROVINCIAS);

  const friendlyNames = {
    conta_corrente: 'Conta Corrente',
    cartao_credito: 'Cartão Crédito',
    plano_pensao: 'Plano Pensão',
    recebimento_recibos: 'Débito Direto',
    conta_nominal: 'Conta Nominal',
    conta_maior_idade: 'Conta Maior Idade',
    conta_terceiros: 'Conta Terceiros',
    conta_particular: 'Conta Particular',
    deposito_prazo_longo: 'Depósito Longo Prazo',
    conta_eletronica: 'Conta Eletrônica',
    fundo_investimento: 'Fundo Investimento',
    valores_mobiliarios: 'Valores Mobiliários',
    deposito_salario: 'Depósito Salário',
    deposito_pensao: 'Depósito Pensão'
  };

  function collectPayload(){
    const genero = document.getElementById('genero').value || null;
    const antiguidade = parseInt(document.getElementById('antiguidade').value, 10);
    const idade = parseInt(document.getElementById('idade').value, 10);
    const renda = parseFloat(document.getElementById('renda').value);
    const segmento = document.getElementById('segmento').value || null;
    const provincia = document.getElementById('provincia').value || null;
    const canal = document.getElementById('canal').value || null;

    const produtos = Array.from(document.querySelectorAll('#produtos input[type="checkbox"][data-product]'))
      .filter(el => el.checked)
      .map(el => el.getAttribute('data-product'));

    return {
      genero: genero || undefined,
      antiguidade: Number.isFinite(antiguidade) ? antiguidade : undefined,
      idade: Number.isFinite(idade) ? idade : undefined,
      renda: Number.isFinite(renda) ? renda : undefined,
      segmento: segmento || undefined,
      provincia: provincia || undefined,
      canal: canal || undefined,
      produtos
    };
  }

  function setLoading(isLoading){
    if(isLoading){
      submitBtn.disabled = true;
      submitBtn.classList.add('opacity-60','cursor-not-allowed');
      feedbackEl.textContent = 'Calculando recomendações...';
    } else {
      submitBtn.disabled = false;
      submitBtn.classList.remove('opacity-60','cursor-not-allowed');
    }
  }

  function renderRanking(ranking){
    rankingBody.innerHTML = '';
    if(!ranking || ranking.length === 0){
      rankingBody.innerHTML = '<tr><td colspan="3" class="p-3 text-sm text-center text-gray-500">Nenhuma recomendação retornada.</td></tr>';
      return;
    }
    ranking.forEach((item, idx) => {
      const tr = document.createElement('tr');
      tr.className = 'border-b border-gray-200 dark:border-gray-700';
      const nome = friendlyNames[item.produto] || item.produto;
      const probRaw = (typeof item.prob === 'number') ? item.prob : parseFloat(item.prob);
      // Multiplicar por 100 para mostrar como porcentagem
      const prob = Number.isFinite(probRaw) ? (probRaw * 100) : null;
      tr.innerHTML = `
        <td class="p-3 text-sm font-medium text-[#111418] dark:text-gray-200">${idx+1}</td>
        <td class="p-3 text-sm text-[#111418] dark:text-gray-200">${nome}</td>
        <td class="p-3 text-sm text-[#111418] dark:text-gray-200 text-right font-mono">${Number.isFinite(prob)? prob.toFixed(3) + '%': '-'}</td>
      `;
      rankingBody.appendChild(tr);
    });
  }

  async function recomendar(payload){
    const resp = await fetch('http://localhost:8000/recomendar',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!resp.ok){
      let msg = 'Erro ao obter recomendações';
      try {
        const err = await resp.json();
        msg = err.detail || JSON.stringify(err);
      } catch(_){}
      throw new Error(msg);
    }
    return resp.json();
  }

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    feedbackEl.textContent = '';
    rankingBody.innerHTML = '';
    const payload = collectPayload();
    setLoading(true);
    try {
      const data = await recomendar(payload);
      renderRanking(data.ranking);
      feedbackEl.textContent = 'Recomendações atualizadas.';
    } catch(err){
      feedbackEl.textContent = 'Falha: ' + err.message;
    } finally {
      setLoading(false);
    }
  });

  form.addEventListener('reset', ()=>{
    feedbackEl.textContent = '';
    rankingBody.innerHTML = '';
  });
})();
