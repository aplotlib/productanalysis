"""
Simplified Dashboard Module for Medical Device Customer Feedback Analyzer

**STABLE & LIGHTWEIGHT VERSION**

Provides essential dashboard functionality with robust error handling
and graceful degradation when optional dependencies are unavailable.

Author: Assistant
Version: 4.0 - Production Stable
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for visualization
def safe_import_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        return go, px, True
    except ImportError:
        logger.warning("Plotly not available - using text-based charts")
        return None, None, False

go, px, PLOTLY_AVAILABLE = safe_import_plotly()

# Professional color scheme
COLORS = {
    'primary': '#1E40AF',
    'secondary': '#059669', 
    'accent': '#DC2626',
    'warning': '#D97706',
    'info': '#0891B2',
    'success': '#22C55E',
    'background': '#F9FAFB'
}

# Risk level colors
RISK_COLORS = {
    'Critical': '🔴',
    'High': '🟠',
    'Medium': '🟡', 
    'Low': '🟢',
    'Minimal': '🟢'
}

class SimpleCharts:
    """Simple chart utilities with text fallbacks"""
    
    @staticmethod
    def create_category_chart(category_data: Dict[str, Any]):
        """Create category breakdown chart"""
        if not PLOTLY_AVAILABLE:
            return SimpleCharts._create_text_chart(category_data)
        
        try:
            # Prepare data
            categories = []
            counts = []
            
            for cat_id, cat_info in category_data.items():
                count = cat_info.get('count', 0)
                if count > 0:
                    categories.append(cat_info.get('name', cat_id))
                    counts.append(count)
            
            if not categories:
                return None
            
            # Create bar chart
            fig = go.Figure(go.Bar(
                y=categories,
                x=counts,
                orientation='h',
                marker=dict(color=COLORS['primary']),
                text=[f"{count} items" for count in counts],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Customer Feedback by Category",
                xaxis_title="Number of Items",
                height=max(300, len(categories) * 40),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return SimpleCharts._create_text_chart(category_data)
    
    @staticmethod
    def _create_text_chart(category_data: Dict[str, Any]) -> str:
        """Create text-based chart as fallback"""
        chart_lines = ["📊 **Category Breakdown:**", ""]
        
        for cat_id, cat_info in category_data.items():
            count = cat_info.get('count', 0)
            if count > 0:
                name = cat_info.get('name', cat_id)
                percentage = cat_info.get('percentage', 0)
                bar = "█" * min(count, 20)  # Visual bar
                chart_lines.append(f"**{name}:** {bar} {count} items ({percentage:.1f}%)")
        
        return "\n".join(chart_lines) if len(chart_lines) > 2 else "No category data available"

class SimpleDashboard:
    """Main dashboard class with essential functionality"""
    
    def __init__(self):
        self.charts = SimpleCharts()
        logger.info("Simple Dashboard initialized")
    
    def render_upload_status(self, processed_data: Dict[str, Any]):
        """Render upload status and summary"""
        if not processed_data or not processed_data.get('success'):
            st.error("❌ No valid data processed")
            return
        
        st.success("✅ Data processed successfully")
        
        # Summary metrics
        summary = processed_data.get('processing_summary', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews", summary.get('valid_reviews', 0))
        
        with col2:
            product_name = summary.get('product_name', 'Unknown')
            st.metric("Product", product_name[:20] + "..." if len(product_name) > 20 else product_name)
        
        with col3:
            asin = summary.get('asin', 'Unknown')
            st.metric("ASIN", asin)
        
        # Export format info
        export_format = processed_data.get('export_format', 'unknown')
        if export_format == 'helium10_reviews':
            st.info("🎯 **Helium 10 Export Detected** - Optimized processing applied")
        elif export_format == 'generic_reviews':
            st.info("📄 **Generic Review Data** - Standard processing applied")
        
        # Data preview
        with st.expander("📄 Data Preview"):
            if 'customer_feedback' in processed_data:
                feedback_data = processed_data['customer_feedback']
                for asin, items in feedback_data.items():
                    if items:
                        st.markdown(f"**{asin}** - {len(items)} feedback items")
                        
                        # Show sample
                        sample_df = pd.DataFrame([
                            {
                                'Date': item.get('date', ''),
                                'Rating': item.get('rating', ''),
                                'Author': item.get('author', ''),
                                'Text Preview': item.get('text', '')[:100] + '...' if item.get('text', '') else ''
                            }
                            for item in items[:5]  # Show first 5
                        ])
                        
                        st.dataframe(sample_df, use_container_width=True)
    
    def render_analysis_overview(self, analysis_results: Dict[str, Any]):
        """Render analysis overview"""
        if not analysis_results:
            st.warning("No analysis results available")
            return
        
        # Get first result if multiple products
        first_result = list(analysis_results.values())[0]
        
        st.markdown("## 📊 Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = first_result.get('total_reviews', 0)
            st.metric("Reviews Analyzed", total_reviews)
        
        with col2:
            quality_assessment = first_result.get('quality_assessment', {})
            quality_score = quality_assessment.get('quality_score', 0)
            st.metric("Quality Score", f"{quality_score:.1f}/100")
        
        with col3:
            risk_level = first_result.get('overall_risk_level', 'Unknown')
            risk_icon = RISK_COLORS.get(risk_level, '⚪')
            st.metric("Risk Level", f"{risk_icon} {risk_level}")
        
        with col4:
            capa_count = len(first_result.get('capa_recommendations', []))
            st.metric("Action Items", capa_count)
        
        # Status indicators
        if quality_score < 60:
            st.warning("⚠️ Quality score below target - review required")
        elif quality_score >= 80:
            st.success("✅ Good quality score - monitor and maintain")
        
        if risk_level in ['Critical', 'High']:
            st.error(f"🚨 {risk_level} risk level - immediate attention required")
        elif risk_level == 'Medium':
            st.warning(f"⚠️ {risk_level} risk level - monitoring recommended")
    
    def render_category_analysis(self, analysis_results: Dict[str, Any]):
        """Render category analysis"""
        # Get first result
        first_result = list(analysis_results.values())[0]
        category_analysis = first_result.get('category_analysis', {})
        
        st.markdown("### Quality Category Breakdown")
        
        # Create chart
        chart = self.charts.create_category_chart(category_analysis)
        
        if PLOTLY_AVAILABLE and chart:
            st.plotly_chart(chart, use_container_width=True)
        elif isinstance(chart, str):
            st.markdown(chart)
        
        # Category details
        st.markdown("### Category Details")
        
        categories_with_issues = []
        for cat_id, cat_data in category_analysis.items():
            count = cat_data.get('count', 0)
            if count > 0:
                categories_with_issues.append({
                    'Category': cat_data.get('name', cat_id),
                    'Count': count,
                    'Percentage': f"{cat_data.get('percentage', 0):.1f}%",
                    'Severity': cat_data.get('severity', 'medium').title(),
                    'Action Required': '✅' if cat_data.get('requires_action', False) else '—'
                })
        
        if categories_with_issues:
            df_categories = pd.DataFrame(categories_with_issues)
            st.dataframe(df_categories, use_container_width=True)
            
            # Show top issues
            st.markdown("#### Top Issues Identified")
            for cat in categories_with_issues[:3]:
                severity_icon = '🔴' if cat['Severity'] == 'Critical' else '🟡'
                st.markdown(f"{severity_icon} **{cat['Category']}**: {cat['Count']} issues ({cat['Percentage']})")
        else:
            st.success("✅ No significant category issues identified")
    
    def render_capa_recommendations(self, analysis_results: Dict[str, Any]):
        """Render CAPA recommendations"""
        # Get first result
        first_result = list(analysis_results.values())[0]
        capa_recommendations = first_result.get('capa_recommendations', [])
        
        st.markdown("### CAPA Recommendations")
        st.caption("Corrective and Preventive Actions for quality improvement")
        
        if not capa_recommendations:
            st.success("✅ No CAPA actions required at this time")
            return
        
        # Group by priority
        priority_groups = {}
        for capa in capa_recommendations:
            priority = capa.get('priority', 'Medium')
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(capa)
        
        # Display by priority
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        priority_icons = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
        
        for priority in priority_order:
            if priority in priority_groups:
                capas = priority_groups[priority]
                icon = priority_icons.get(priority, '⚪')
                
                st.markdown(f"#### {icon} {priority} Priority ({len(capas)} items)")
                
                for i, capa in enumerate(capas, 1):
                    with st.expander(f"{priority}-{i}: {capa.get('category', 'General')}"):
                        # Main content
                        st.markdown(f"**Issue:** {capa.get('issue_description', 'Not specified')}")
                        st.markdown(f"**Corrective Action:** {capa.get('corrective_action', 'Not specified')}")
                        st.markdown(f"**Timeline:** {capa.get('timeline', 'Not specified')}")
                        st.markdown(f"**Responsibility:** {capa.get('responsibility', 'Not specified')}")
                        
                        # Additional details
                        if capa.get('affected_customers'):
                            st.markdown(f"**Affected Customers:** {capa['affected_customers']}")
                        
                        if capa.get('success_metrics'):
                            metrics = capa['success_metrics']
                            if isinstance(metrics, list):
                                st.markdown("**Success Metrics:**")
                                for metric in metrics:
                                    st.markdown(f"• {metric}")
    
    def render_ai_insights(self, analysis_results: Dict[str, Any]):
        """Render AI insights if available"""
        # Get first result
        first_result = list(analysis_results.values())[0]
        ai_insights = first_result.get('ai_insights', {})
        ai_available = first_result.get('ai_analysis_available', False)
        
        st.markdown("### AI-Powered Insights")
        
        if not ai_available:
            st.info("🤖 AI analysis not available. Configure OpenAI API key for enhanced insights.")
            return
        
        if not ai_insights:
            st.warning("AI analysis was attempted but no insights were generated")
            return
        
        # Overall sentiment
        if 'overall_sentiment' in ai_insights:
            sentiment = ai_insights['overall_sentiment']
            st.markdown(f"**Overall Sentiment:** {sentiment}")
        
        # Safety concerns (most important)
        if 'safety_concerns' in ai_insights:
            safety_concerns = ai_insights['safety_concerns']
            if safety_concerns:
                st.markdown("#### 🚨 Safety Concerns")
                if isinstance(safety_concerns, list):
                    for concern in safety_concerns:
                        st.error(f"⚠️ {concern}")
                else:
                    st.error(f"⚠️ {safety_concerns}")
        
        # Top quality issues
        if 'top_quality_issues' in ai_insights:
            quality_issues = ai_insights['top_quality_issues']
            if quality_issues:
                st.markdown("#### 🔍 Top Quality Issues")
                if isinstance(quality_issues, list):
                    for issue in quality_issues:
                        st.markdown(f"• {issue}")
                else:
                    st.markdown(f"• {quality_issues}")
        
        # Immediate actions
        if 'immediate_actions' in ai_insights:
            actions = ai_insights['immediate_actions']
            if actions:
                st.markdown("#### ⚡ Immediate Actions Needed")
                if isinstance(actions, list):
                    for action in actions:
                        st.warning(f"🎯 {action}")
                else:
                    st.warning(f"🎯 {actions}")
        
        # Listing improvements
        if 'listing_improvements' in ai_insights:
            improvements = ai_insights['listing_improvements']
            if improvements:
                st.markdown("#### 📝 Listing Improvements")
                st.info(improvements)
    
    def render_export_options(self, analysis_results: Dict[str, Any]):
        """Render export options"""
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Summary Report", use_container_width=True):
                self._export_summary_report(analysis_results)
        
        with col2:
            if st.button("📋 Export CAPA List", use_container_width=True):
                self._export_capa_list(analysis_results)
    
    def _export_summary_report(self, analysis_results: Dict[str, Any]):
        """Export summary report"""
        try:
            # Get first result
            first_result = list(analysis_results.values())[0]
            
            # Create report content
            report_lines = [
                "# CUSTOMER FEEDBACK ANALYSIS REPORT",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Product:** {first_result.get('product_name', 'Unknown')}",
                f"**ASIN:** {first_result.get('asin', 'Unknown')}",
                f"**Total Reviews:** {first_result.get('total_reviews', 0)}",
                "",
                "## QUALITY SUMMARY",
                f"**Quality Score:** {first_result.get('quality_assessment', {}).get('quality_score', 0):.1f}/100",
                f"**Risk Level:** {first_result.get('overall_risk_level', 'Unknown')}",
                "",
                "## KEY FINDINGS"
            ]
            
            # Add category summary
            category_analysis = first_result.get('category_analysis', {})
            for cat_id, cat_data in category_analysis.items():
                count = cat_data.get('count', 0)
                if count > 0:
                    report_lines.append(f"- **{cat_data.get('name', cat_id)}:** {count} issues")
            
            # Add CAPA summary
            capa_recommendations = first_result.get('capa_recommendations', [])
            if capa_recommendations:
                report_lines.extend(["", "## ACTION ITEMS"])
                for capa in capa_recommendations:
                    priority = capa.get('priority', 'Medium')
                    category = capa.get('category', 'General')
                    report_lines.append(f"- **{priority}:** {category}")
            
            report_text = "\n".join(report_lines)
            
            # Create download
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            st.error(f"Export failed: {str(e)}")
    
    def _export_capa_list(self, analysis_results: Dict[str, Any]):
        """Export CAPA list"""
        try:
            # Get first result
            first_result = list(analysis_results.values())[0]
            capa_recommendations = first_result.get('capa_recommendations', [])
            
            if not capa_recommendations:
                st.warning("No CAPA items to export")
                return
            
            # Create CAPA CSV data
            capa_data = []
            for i, capa in enumerate(capa_recommendations, 1):
                capa_data.append({
                    'CAPA_ID': f"CAPA-{i:03d}",
                    'Priority': capa.get('priority', 'Medium'),
                    'Category': capa.get('category', 'General'),
                    'Issue': capa.get('issue_description', ''),
                    'Corrective_Action': capa.get('corrective_action', ''),
                    'Timeline': capa.get('timeline', ''),
                    'Responsibility': capa.get('responsibility', ''),
                    'Status': 'Open'
                })
            
            # Convert to CSV
            df = pd.DataFrame(capa_data)
            csv = df.to_csv(index=False)
            
            # Create download
            st.download_button(
                label="📥 Download CAPA List",
                data=csv,
                file_name=f"CAPA_List_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            logger.error(f"Error exporting CAPA list: {str(e)}")
            st.error(f"Export failed: {str(e)}")

# Export main class
__all__ = ['SimpleDashboard']
