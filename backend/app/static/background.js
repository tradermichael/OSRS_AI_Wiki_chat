(() => {
  const canvas = document.getElementById('bg');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const DPR = Math.min(2, window.devicePixelRatio || 1);

  let w = 0;
  let h = 0;

  function resize() {
    w = Math.max(1, window.innerWidth);
    h = Math.max(1, window.innerHeight);
    canvas.width = Math.floor(w * DPR);
    canvas.height = Math.floor(h * DPR);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  const rand = (a, b) => a + Math.random() * (b - a);

  const particles = Array.from({ length: 90 }, () => ({
    x: rand(0, window.innerWidth),
    y: rand(0, window.innerHeight),
    r: rand(0.6, 2.2),
    vx: rand(-0.12, 0.12),
    vy: rand(-0.06, 0.10),
    a: rand(0.10, 0.60),
    tw: rand(0.002, 0.010),
    t: rand(0, 1000),
    hue: rand(38, 52) // warm gold
  }));

  let last = performance.now();

  function draw(ts) {
    const dt = Math.min(0.05, (ts - last) / 1000);
    last = ts;

    // Backdrop: deep night + subtle vignette
    ctx.clearRect(0, 0, w, h);

    const g = ctx.createRadialGradient(w * 0.22, h * 0.10, 0, w * 0.5, h * 0.6, Math.max(w, h));
    g.addColorStop(0, 'rgba(33, 46, 72, 0.75)');
    g.addColorStop(0.55, 'rgba(14, 20, 34, 0.92)');
    g.addColorStop(1, 'rgba(6, 8, 12, 0.98)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    // Rune-ish swirls (soft arcs)
    ctx.globalAlpha = 0.12;
    ctx.strokeStyle = 'rgba(194, 157, 66, 0.55)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 6; i++) {
      const cx = w * (0.10 + i * 0.16);
      const cy = h * (0.20 + (i % 2) * 0.08);
      const r = Math.min(w, h) * (0.18 + i * 0.03);
      ctx.beginPath();
      ctx.arc(cx, cy, r, rand(0, Math.PI), rand(Math.PI, Math.PI * 2));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Particles
    for (const p of particles) {
      p.t += dt;
      p.x += p.vx * 60 * dt;
      p.y += p.vy * 60 * dt;

      if (p.x < -20) p.x = w + 20;
      if (p.x > w + 20) p.x = -20;
      if (p.y < -20) p.y = h + 20;
      if (p.y > h + 20) p.y = -20;

      const twinkle = 0.65 + 0.35 * Math.sin(p.t / (p.tw * 1000));
      const alpha = p.a * twinkle;

      ctx.beginPath();
      ctx.fillStyle = `hsla(${p.hue}, 65%, 65%, ${alpha})`;
      ctx.shadowColor = `hsla(${p.hue}, 70%, 65%, ${alpha})`;
      ctx.shadowBlur = 10;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.shadowBlur = 0;
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(draw);
})();
