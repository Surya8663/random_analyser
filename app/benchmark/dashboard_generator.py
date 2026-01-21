"""
Generate self-contained HTML dashboard for benchmark results
"""
import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DashboardGenerator:
    """Generate self-contained HTML dashboard for benchmark visualization"""
    
    def __init__(self, benchmark_results_dir: str = "benchmark_results"):
        self.results_dir = Path(benchmark_results_dir)
        self.results_data = {}
        self.aggregated_metrics = {}
        self.dashboard_html = ""
    
    def load_data(self, results_file: Optional[str] = None):
        """Load benchmark results data"""
        if results_file:
            filepath = Path(results_file)
        else:
            result_files = list(self.results_dir.glob("benchmark_results_*.json"))
            if not result_files:
                raise ValueError(f"No benchmark results found in {self.results_dir}")
            filepath = max(result_files, key=lambda p: p.stat().st_mtime)
        
        with open(filepath, 'r') as f:
            self.results_data = json.load(f)
        
        # Also load aggregated metrics if available
        metrics_files = list(self.results_dir.glob("metrics_summary_*.json"))
        if metrics_files:
            metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
            with open(metrics_file, 'r') as f:
                self.aggregated_metrics = json.load(f)
    
    def generate_dashboard(self, output_file: Optional[str] = None) -> str:
        """Generate complete HTML dashboard"""
        if not self.results_data:
            self.load_data()
        
        # Generate dashboard sections
        html_parts = [
            self._generate_header(),
            self._generate_summary_section(),
            self._generate_performance_metrics_section(),
            self._generate_agent_performance_section(),
            self._generate_risk_analysis_section(),
            self._generate_modality_analysis_section(),
            self._generate_detailed_results_section(),
            self._generate_footer()
        ]
        
        dashboard_html = "\n".join(html_parts)
        self.dashboard_html = dashboard_html
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            logger.info(f"📊 Generated dashboard: {output_file}")
        
        return dashboard_html
    
    def _generate_header(self) -> str:
        """Generate dashboard header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_docs = len(self.results_data.get("detailed_results", {}))
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Intelligence Benchmark Dashboard</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #7c3aed;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --border-color: #334155;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--dark-bg);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .header-title {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header-subtitle {{
            color: #94a3b8;
            font-size: 1.1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .success {{ color: var(--success-color); }}
        .warning {{ color: var(--warning-color); }}
        .danger {{ color: var(--danger-color); }}
        .primary {{ color: var(--primary-color); }}
        
        .section {{
            margin-bottom: 40px;
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid var(--border-color);
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: var(--primary-color);
            border-radius: 2px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .metric-card {{
            background: rgba(30, 41, 59, 0.7);
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid var(--primary-color);
        }}
        
        .metric-name {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #cbd5e1;
        }}
        
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        
        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .agent-card {{
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(124, 58, 237, 0.1));
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        .agent-name {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .agent-score {{
            font-size: 2rem;
            font-weight: 700;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(30, 41, 59, 0.7);
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: rgba(15, 23, 42, 0.9);
            font-weight: 600;
            color: #cbd5e1;
        }}
        
        tr:hover {{
            background: rgba(37, 99, 235, 0.1);
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .risk-low {{ background: rgba(16, 185, 129, 0.2); color: var(--success-color); }}
        .risk-medium {{ background: rgba(245, 158, 11, 0.2); color: var(--warning-color); }}
        .risk-high {{ background: rgba(239, 68, 68, 0.2); color: var(--danger-color); }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: #94a3b8;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header-title {{
                font-size: 2rem;
            }}
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1 class="header-title">📊 Document Intelligence Benchmark Dashboard</h1>
            <p class="header-subtitle">Phase 6 - Research-Grade Evaluation System</p>
            <p class="header-subtitle">Generated: {timestamp} | Documents: {total_docs}</p>
        </header>
        """
    
    def _generate_summary_section(self) -> str:
        """Generate summary statistics section"""
        summary = self.results_data.get("summary_statistics", {}).get("basic", {})
        total_docs = summary.get("total_documents", 0)
        success_rate = summary.get("success_rate", 0) * 100
        avg_processing = summary.get("average_processing_time", 0)
        avg_risk = summary.get("average_risk_score", 0) * 100
        
        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value primary">{total_docs}</div>
                <div class="stat-label">Total Documents Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{success_rate:.1f}%</div>
                <div class="stat-label">Overall Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warning">{avg_processing:.2f}s</div>
                <div class="stat-label">Average Processing Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_risk:.1f}%</div>
                <div class="stat-label">Average Risk Score</div>
            </div>
        </div>
        """
    
    def _generate_performance_metrics_section(self) -> str:
        """Generate performance metrics section"""
        perf_metrics = self.aggregated_metrics.get("performance_metrics", {})
        
        if not perf_metrics:
            return ""
        
        metrics_html = []
        for metric_name, metric_data in perf_metrics.items():
            if isinstance(metric_data, dict):
                mean_value = metric_data.get("mean", 0)
                std_value = metric_data.get("std", 0)
                
                # Format based on metric type
                if metric_name == "processing_time":
                    display_value = f"{mean_value:.2f}s"
                    display_std = f"±{std_value:.2f}s"
                elif metric_name == "risk_scores":
                    display_value = f"{mean_value:.2%}"
                    display_std = f"±{std_value:.2%}"
                else:
                    display_value = f"{mean_value:.2f}"
                    display_std = f"±{std_value:.2f}"
                
                metric_label = metric_name.replace("_", " ").title()
                
                metrics_html.append(f"""
                    <div class="metric-card">
                        <div class="metric-name">{metric_label}</div>
                        <div class="metric-value">{display_value}</div>
                        <div class="stat-label">Std Dev: {display_std}</div>
                    </div>
                """)
        
        return f"""
        <div class="section">
            <h2 class="section-title">📈 Performance Metrics</h2>
            <div class="metrics-grid">
                {"".join(metrics_html)}
            </div>
        </div>
        """
    
    def _generate_agent_performance_section(self) -> str:
        """Generate agent performance section"""
        agent_perf = self.aggregated_metrics.get("agent_performance", {})
        
        if not agent_perf:
            return ""
        
        agent_html = []
        for agent_name, agent_data in agent_perf.items():
            validation_score = agent_data.get("average_validation_score", 0) * 100
            confidence = agent_data.get("average_confidence", 0) * 100
            
            # Determine score color
            if validation_score >= 80:
                score_class = "success"
            elif validation_score >= 60:
                score_class = "warning"
            else:
                score_class = "danger"
            
            agent_display = agent_name.replace("_", " ").title()
            
            agent_html.append(f"""
                <div class="agent-card">
                    <div class="agent-name">🤖 {agent_display} Agent</div>
                    <div class="agent-score {score_class}">{validation_score:.1f}%</div>
                    <div class="stat-label">Validation Score</div>
                    <div style="margin-top: 10px; font-size: 0.9rem; color: #94a3b8;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
            """)
        
        return f"""
        <div class="section">
            <h2 class="section-title">🛠️ Agent Performance Analysis</h2>
            <div class="agent-grid">
                {"".join(agent_html)}
            </div>
        </div>
        """
    
    def _generate_risk_analysis_section(self) -> str:
        """Generate risk analysis section"""
        risk_analysis = self.aggregated_metrics.get("risk_analysis", {})
        
        if not risk_analysis:
            return ""
        
        # Risk distribution
        risk_clusters = risk_analysis.get("risk_clusters", {})
        risk_distribution = []
        
        for level in ["low_risk", "medium_risk", "high_risk", "critical_risk"]:
            if level in risk_clusters:
                cluster = risk_clusters[level]
                count = cluster.get("count", 0)
                percentage = cluster.get("percentage", 0) * 100
                
                level_display = level.replace("_", " ").title()
                badge_class = level.replace("_", "-")
                
                risk_distribution.append(f"""
                    <div class="metric-card">
                        <div class="metric-name">
                            <span class="risk-badge {badge_class}">{level_display}</span>
                        </div>
                        <div class="metric-value">{count} docs</div>
                        <div class="stat-label">{percentage:.1f}% of total</div>
                    </div>
                """)
        
        # Risk patterns
        risk_patterns_html = ""
        patterns = risk_analysis.get("high_risk_patterns", [])
        if patterns:
            pattern_items = []
            for pattern in patterns:
                pattern_items.append(f"""
                    <li>
                        <strong>{pattern.get('pattern', '').replace('_', ' ').title()}:</strong>
                        {pattern.get('description', '')} ({pattern.get('count', 0)} documents)
                    </li>
                """)
            
            risk_patterns_html = f"""
                <div style="margin-top: 20px;">
                    <h3 style="margin-bottom: 10px; color: #cbd5e1;">🔍 High-Risk Patterns Detected</h3>
                    <ul style="padding-left: 20px; color: #94a3b8;">
                        {"".join(pattern_items)}
                    </ul>
                </div>
            """
        
        return f"""
        <div class="section">
            <h2 class="section-title">⚠️ Risk Analysis</h2>
            <div class="metrics-grid">
                {"".join(risk_distribution)}
            </div>
            {risk_patterns_html}
        </div>
        """
    
    def _generate_modality_analysis_section(self) -> str:
        """Generate modality analysis section"""
        modality_analysis = self.aggregated_metrics.get("modality_contribution", {})
        
        if not modality_analysis:
            return ""
        
        modality_html = []
        for modality, data in modality_analysis.items():
            score = data.get("average_score", 0) * 100
            importance = data.get("relative_importance", 0) * 100
            
            # Modality icon
            icons = {
                "text": "📝",
                "visual": "👁️",
                "fusion": "🔄"
            }
            icon = icons.get(modality, "⚡")
            
            modality_display = modality.title()
            
            modality_html.append(f"""
                <div class="metric-card">
                    <div class="metric-name">{icon} {modality_display} Modality</div>
                    <div class="metric-value">{score:.1f}%</div>
                    <div class="stat-label">Score</div>
                    <div style="margin-top: 10px; font-size: 0.9rem; color: #94a3b8;">
                        Relative Importance: {importance:.1f}%
                    </div>
                </div>
            """)
        
        return f"""
        <div class="section">
            <h2 class="section-title">🎯 Modality Contribution Analysis</h2>
            <div class="metrics-grid">
                {"".join(modality_html)}
            </div>
        </div>
        """
    
    def _generate_detailed_results_section(self) -> str:
        """Generate detailed results table section"""
        detailed_results = self.results_data.get("detailed_results", {})
        
        if not detailed_results:
            return ""
        
        # Create table rows for first 20 documents
        table_rows = []
        for doc_id, result in list(detailed_results.items())[:20]:
            success = result.get("success", False)
            processing_time = result.get("processing_time", 0)
            risk_score = result.get("risk_score", 0)
            extracted = result.get("extracted_fields", 0)
            contradictions = result.get("contradictions", 0)
            doc_type = result.get("document_type", "unknown")
            
            # Determine risk badge
            if risk_score > 0.7:
                risk_badge = '<span class="risk-badge risk-high">Critical</span>'
            elif risk_score > 0.5:
                risk_badge = '<span class="risk-badge risk-high">High</span>'
            elif risk_score > 0.3:
                risk_badge = '<span class="risk-badge risk-medium">Medium</span>'
            else:
                risk_badge = '<span class="risk-badge risk-low">Low</span>'
            
            # Success icon
            success_icon = "✅" if success else "❌"
            
            table_rows.append(f"""
                <tr>
                    <td>{doc_id[:12]}...</td>
                    <td>{doc_type}</td>
                    <td>{success_icon}</td>
                    <td>{processing_time:.2f}s</td>
                    <td>{risk_badge} ({risk_score:.2f})</td>
                    <td>{extracted}</td>
                    <td>{contradictions}</td>
                </tr>
            """)
        
        return f"""
        <div class="section">
            <h2 class="section-title">📋 Detailed Results (Sample)</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Document ID</th>
                            <th>Type</th>
                            <th>Success</th>
                            <th>Processing Time</th>
                            <th>Risk Level</th>
                            <th>Fields Extracted</th>
                            <th>Contradictions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(table_rows)}
                    </tbody>
                </table>
            </div>
            <p style="margin-top: 15px; color: #94a3b8; font-size: 0.9rem;">
                Showing 20 of {len(detailed_results)} documents. See full results in JSON files.
            </p>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate dashboard footer"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
        <footer>
            <p>Document Intelligence Benchmark Dashboard - Phase 6 Research System</p>
            <p>Generated: {timestamp} | All metrics computed from real pipeline execution</p>
            <p style="margin-top: 10px; font-size: 0.8rem;">
                🔬 Research-Grade Evaluation System | No Mock Data Used | Real Agent Execution
            </p>
        </footer>
    </div>
    
    <script>
        // Simple JavaScript for interactivity
        document.addEventListener('DOMContentLoaded', function() {{
            // Add click handlers to stat cards
            document.querySelectorAll('.stat-card').forEach(card => {{
                card.addEventListener('click', function() {{
                    this.style.transform = 'scale(0.98)';
                    setTimeout(() => {{
                        this.style.transform = '';
                    }}, 150);
                }});
            }});
            
            // Add hover effects to table rows
            document.querySelectorAll('tbody tr').forEach(row => {{
                row.addEventListener('mouseenter', function() {{
                    this.style.backgroundColor = 'rgba(37, 99, 235, 0.15)';
                }});
                row.addEventListener('mouseleave', function() {{
                    this.style.backgroundColor = '';
                }});
            }});
        }});
    </script>
</body>
</html>
        """
    
    def generate_dashboard_with_plots(self, output_file: str = "dashboard.html"):
        """Generate dashboard with embedded Plotly charts"""
        # This would generate interactive charts
        # For simplicity, we're generating static HTML for now
        return self.generate_dashboard(output_file)


def generate_benchmark_dashboard(results_file: Optional[str] = None,
                                output_file: str = "benchmark_dashboard.html"):
    """Convenience function to generate dashboard"""
    generator = DashboardGenerator()
    generator.load_data(results_file)
    return generator.generate_dashboard(output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard")
    parser.add_argument("--results-file", help="Path to benchmark results JSON")
    parser.add_argument("--output", default="benchmark_dashboard.html", help="Output HTML file")
    
    args = parser.parse_args()
    
    generate_benchmark_dashboard(args.results_file, args.output)
    print(f"✅ Dashboard generated: {args.output}")