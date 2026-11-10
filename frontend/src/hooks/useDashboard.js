import { useEffect, useRef, useState } from 'react';

export function useDashboard(api) {
  const [attempt, setAttempt] = useState(0);
  const [boot, setBoot] = useState({ status: 'loading', metadata: null, error: null });
  const [selection, setSelection] = useState(null);
  const [replay, setReplay] = useState({ status: 'idle', data: null, error: null });
  const generation = useRef(0);
  useEffect(() => {
    const controller = new AbortController(); let active = true;
    setBoot({ status: 'loading', metadata: null, error: null }); setSelection(null);
    setReplay({ status: 'idle', data: null, error: null });
    (async () => {
      try {
        const [, metadata] = await Promise.all([api.health(controller.signal), api.metadata(controller.signal)]);
        if (!active) return;
        if (!metadata.sensor_details.length || !metadata.replay.available || !metadata.supported_horizons.length) {
          setBoot({ status: 'empty', metadata, error: null }); return;
        }
        setBoot({ status: 'ready', metadata, error: null });
        setSelection({ sensor_id: metadata.sensor_details[0].sensor_id, index: metadata.replay.first_index, scenario: 'clean', policy: 'controller' });
      } catch (error) { if (active && error.name !== 'AbortError') setBoot({ status: 'error', metadata: null, error }); }
    })();
    return () => { active = false; controller.abort(); };
  }, [api, attempt]);

  useEffect(() => {
    if (!selection || boot.status !== 'ready') return;
    const controller = new AbortController(); const current = ++generation.current;
    setReplay({ status: 'loading', data: null, error: null });
    api.replay(selection, controller.signal).then(data => {
      if (!controller.signal.aborted && generation.current === current) setReplay({ status: 'ready', data, error: null });
    }).catch(error => {
      if (!controller.signal.aborted && generation.current === current && error.name !== 'AbortError') setReplay({ status: 'error', data: null, error });
    });
    return () => controller.abort();
  }, [api, selection, boot.status]);
  function change(patch) {
    // Clear old forecasts synchronously so a new selection never displays old values.
    generation.current += 1; setReplay({ status: 'loading', data: null, error: null });
    setSelection(value => ({ ...value, ...patch }));
  }
  return { boot, selection, replay, change, retry: () => setAttempt(value => value + 1) };
}
