const cases = [
  { label: 'Clean replay', scenario: 'clean', policy: 'controller' },
  { label: 'Missing inputs', scenario: 'missing', policy: 'controller' },
  { label: 'Noisy inputs', scenario: 'noisy', policy: 'controller' },
  { label: 'Stuck inputs', scenario: 'stuck', policy: 'controller' },
  { label: 'Clean primary intervals', scenario: 'clean', policy: 'xgboost' },
];

export function DemoGuide({ metadata, change }) {
  return <details className="panel demo-guide">
    <summary>Demo walkthrough <span>Five cases to explore</span></summary>
    <div className="guide-content">
      <p>Choose a case at the first available replay origin for your selected sensor. Each case requests actual backend forecasts; model choices and results are not predetermined by this guide.</p>
      <div className="guide-actions">{cases.filter(item => metadata.scenarios.includes(item.scenario) && metadata.policies.includes(item.policy)).map(item => <button className="secondary" key={item.label} onClick={() => change({ index: metadata.replay.first_index, scenario: item.scenario, policy: item.policy })}>{item.label}</button>)}</div>
      <dl className="guide-notes">
        <div><dt>Read the timeline</dt><dd>Twelve past readings give context. Diamonds predict +15, +30 and +60 minutes from the replay origin; gaps are unavailable inputs, not measured zero speeds.</dd></div>
        <div><dt>Inspect the decision</dt><dd>One backend controller decision serves every sensor and horizon. Quality indicators are not confirmed physical faults; seasonal fallback is not sensor repair.</dd></div>
        <div><dt>Interpret intervals carefully</dt><dd>Only clean fixed-XGBoost calibration bounds may appear. They are not independently validated; controller, fallback and faulty-input intervals are unavailable.</dd></div>
      </dl>
    </div>
  </details>;
}
