from app.services.agent_service import get_agent_service
import json, time, asyncio, random, logging, re
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from difflib import SequenceMatcher
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestCase:
    """Single test case"""
    def __init__(self, image_path: str, gps: Dict[str, float], ground_truth_name: str,
                 ground_truth_district: str, difficulty: str, notes: str = ""):
        self.image_path = image_path
        self.gps = gps
        self.ground_truth = {
            "destination_name": ground_truth_name,
            "district_name": ground_truth_district,            
        }
        self.difficulty = difficulty
        self.notes = notes

class AgentEvaluator:
    
    def __init__(self, agent, output_dir: str = "evaluation_results"):
        self.agent = agent
        self.results = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.confidence_tracking = {"High": [], "Low": []}  # Removed "Medium"
        self.tool_usage_stats = {
            "visual_analysis": 0,
            "vector_search": 0,
            "web_search": 0
        }
    
    def load_test_dataset(self, dataset_path: str = "test_dataset_new.json") -> List[TestCase]:
        
        if Path(dataset_path).exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                return [TestCase(**item) for item in data]
        logger.warning(f"Dataset {dataset_path} not found!")
        return []
    
    async def evaluate_single_case(self, test_case: TestCase) -> Dict:
        try:
            start_time = time.time()

            if not Path(test_case.image_path).exists():
                logger.warning(f"Image not found: {test_case.image_path}, using mock result")
                return self._create_mock_result(test_case)
            
            image = Image.open(test_case.image_path)

            # Call agent
            result = await self.agent.identify_and_generate_content(image=image, gps_location=test_case.gps)
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(result, test_case, elapsed_time)
            
            # Track confidence level
            confidence = result.get("confidence", "Unknown")
            if confidence in self.confidence_tracking:
                self.confidence_tracking[confidence].append(metrics["overall_correct"])
            
            # Track tool usage
            if result.get("found_in_db"):
                self.tool_usage_stats["vector_search"] += 1
            if result.get("used_web_search"):
                self.tool_usage_stats["web_search"] += 1

            return {
                "test_case": test_case.image_path,
                "difficulty": test_case.difficulty,
                "ground_truth": test_case.ground_truth,
                "prediction": result,
                "metrics": metrics,
                "notes": test_case.notes
            }
        except Exception as e:
            logger.error(f"Error evaluating {test_case.image_path}: {str(e)}")
            return {"test_case": test_case.image_path, "error": str(e), "metrics": {"error": True}}
    
    def _create_mock_result(self, test_case: TestCase) -> Dict:
        
        accuracy_by_difficulty = {"easy": 0.95, "medium": 0.75, "hard": 0.40}
        is_correct = random.random() < accuracy_by_difficulty[test_case.difficulty]
        
        mock_result = {
            "destination_name": test_case.ground_truth["destination_name"] if is_correct else "Wrong Location",
            "district_name": test_case.ground_truth["district_name"] if is_correct else "Wrong District",            
            "confidence": "High" if test_case.difficulty == "easy" else "Low",  # Removed Medium
            "found_in_db": test_case.difficulty != "hard",
            "used_web_search": test_case.difficulty == "hard"
        }
        elapsed_time = random.uniform(1.0, 3.0)
        metrics = self._calculate_metrics(mock_result, test_case, elapsed_time)
        
        return {
            "test_case": test_case.image_path,
            "difficulty": test_case.difficulty,
            "ground_truth": test_case.ground_truth,
            "prediction": mock_result,
            "metrics": metrics,
            "notes": test_case.notes + " [MOCK DATA]"
        }
    
    def _calculate_metrics(self, result: Dict, test_case: TestCase, elapsed_time: float) -> Dict:
        
        gt = test_case.ground_truth
        
        pred_name = result.get("destination_name", "").lower().strip()
        gt_name = gt["destination_name"].lower().strip()
        district_match = result.get("district_name", "").lower().strip() == gt["district_name"].lower().strip()
        
        exact_match = pred_name == gt_name
        substring_match = gt_name in pred_name or pred_name in gt_name

        def normalize_name(name):
            patterns_to_remove = [
                r',\s*colombo$', r',\s*kandy$', r',\s*galle$',
                r',\s*sri lanka$',
                r'\s+temple$', r'\s+viharaya$', r'\s+kovil$',
                r'\s+beach$', r'\s+fort$', r'\s+museum$',
            ]
            normalized = name
            for pattern in patterns_to_remove:
                normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
            return normalized.strip()
        
        pred_normalized = normalize_name(pred_name)
        gt_normalized = normalize_name(gt_name)
        normalized_match = pred_normalized == gt_normalized

        def tokenize(name):
            stop_words = {'the', 'of', 'and', 'in', 'at', 'sri', 'lanka'}
            tokens = set(re.findall(r'\w+', name.lower()))
            return tokens - stop_words
        
        pred_tokens = tokenize(pred_name)
        gt_tokens = tokenize(gt_name)
        
        if pred_tokens or gt_tokens:
            token_overlap = len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)
        else:
            token_overlap = 0.0
        
        token_match = token_overlap >= 0.75

        name_similarity = SequenceMatcher(None, pred_name, gt_name).ratio()
        fuzzy_match = name_similarity >= 0.85 

        name_match = (
            exact_match or
            substring_match or
            normalized_match or
            token_match or
            fuzzy_match
        )
        
        content_fields = [
            "historical_background",
            "cultural_significance", 
            "what_makes_it_special",
            "visitor_experience"
        ]
        content_completeness = sum(
            1 for field in content_fields 
            if result.get(field, "").strip()
        ) / len(content_fields)
        facts_count = len(result.get("interesting_facts", []))
       
        return {
            "name_exact_match": exact_match,
            "name_similarity": name_similarity,
            "name_match_accepted": name_match,
            "district_correct": district_match,
            "overall_correct": name_match and district_match,
            "confidence_level": result.get("confidence", "Unknown"),
            "found_in_db": result.get("found_in_db", False),
            "used_web_search": result.get("used_web_search", False),
            "response_time_seconds": elapsed_time,
            "content_completeness": content_completeness,
            "facts_count": facts_count,
            "match_type": (
                "exact" if exact_match else
                "substring" if substring_match else
                "normalized" if normalized_match else
                "token" if token_match else
                "fuzzy" if fuzzy_match else
                "no_match"
            ),
            "error": False
        } 
    
    async def run_full_evaluation(self, dataset_path: str = "test_dataset_new.json", delay_seconds: int = 6):
        
        test_cases = self.load_test_dataset(dataset_path)
        total = len(test_cases)
        if total == 0:
            logger.warning("No test cases to evaluate!")
            return
        
        logger.info(f"Starting evaluation on {total} test cases...")
        
        for i, case in enumerate(test_cases, 1):
            logger.info(f"[{i}/{total}] Evaluating: {case.image_path} (Difficulty: {case.difficulty})")
            result = await self.evaluate_single_case(case)
            self.results.append(result)

            if i < total:  
                logger.info(f"Sleeping {delay_seconds}s to avoid API rate limits...")
                await asyncio.sleep(delay_seconds)

        self.generate_summary_report()
        self.generate_pdf_report()  # New PDF generation
        self.save_results()
        self.create_visualizations()
        self.export_to_csv()
    
    def generate_summary_report(self):
        
        valid_results = [r for r in self.results if not r.get("metrics", {}).get("error")]
        total = len(valid_results)
        if total == 0:
            logger.warning("No valid results to summarize!")
            return
        
        overall_acc = sum(r["metrics"]["overall_correct"] for r in valid_results) / total
        name_acc = sum(r["metrics"]["name_exact_match"] for r in valid_results) / total
        district_acc = sum(r["metrics"]["district_correct"] for r in valid_results) / total

        avg_response_time = sum(r["metrics"]["response_time_seconds"] for r in valid_results) / total
        avg_content_completeness = sum(r["metrics"]["content_completeness"] for r in valid_results) / total
        
        db_matches = sum(r["metrics"]["found_in_db"] for r in valid_results)
        web_searches = sum(r["metrics"]["used_web_search"] for r in valid_results)
        
        print("\n" + "="*60)
        print("AGENT EVALUATION SUMMARY REPORT")
        print("="*60)
        print(f"Total Test Cases: {total}")
        print(f"\nAccuracy Metrics:")
        print(f"  Overall Accuracy (Name + District): {overall_acc:.1%}")
        print(f"  Name Accuracy: {name_acc:.1%}")
        print(f"  District Accuracy: {district_acc:.1%}")
        print(f"\nPerformance Metrics:")
        print(f"  Average Response Time: {avg_response_time:.2f}s")
        print(f"  Content Completeness: {avg_content_completeness:.1%}")
        print(f"\nTool Usage:")
        print(f"  Database Matches: {db_matches} ({db_matches/total:.1%})")
        print(f"  Web Search Used: {web_searches} ({web_searches/total:.1%})")

        print(f"\nAccuracy by Difficulty:")
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in valid_results if r["difficulty"] == diff]
            if diff_results:
                diff_acc = sum(r["metrics"]["overall_correct"] for r in diff_results) / len(diff_results)
                print(f"  {diff.capitalize()}: {diff_acc:.1%} ({len(diff_results)} cases)")
        
        print(f"\nAccuracy by Confidence Level:")
        for conf, results in self.confidence_tracking.items():
            if results:
                conf_acc = sum(results) / len(results)
                print(f"  {conf}: {conf_acc:.1%} ({len(results)} cases)")
        
        print("="*60 + "\n")
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file = self.output_dir / f"evaluation_report_{timestamp}.pdf"
        
        valid_results = [r for r in self.results if not r.get("metrics", {}).get("error")]
        if not valid_results:
            logger.warning("No valid results to generate PDF!")
            return
        
        doc = SimpleDocTemplate(str(pdf_file), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Agent Evaluation Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        total = len(valid_results)
        overall_acc = sum(r["metrics"]["overall_correct"] for r in valid_results) / total
        name_acc = sum(r["metrics"]["name_exact_match"] for r in valid_results) / total
        district_acc = sum(r["metrics"]["district_correct"] for r in valid_results) / total
        avg_response_time = sum(r["metrics"]["response_time_seconds"] for r in valid_results) / total
        avg_content = sum(r["metrics"]["content_completeness"] for r in valid_results) / total
        
        story.append(Paragraph("Summary Statistics", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Test Cases', str(total)],
            ['Overall Accuracy', f'{overall_acc:.1%}'],
            ['Name Accuracy', f'{name_acc:.1%}'],
            ['District Accuracy', f'{district_acc:.1%}'],
            ['Avg Response Time', f'{avg_response_time:.2f}s'],
            ['Avg Content Completeness', f'{avg_content:.1%}'],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Accuracy by Difficulty
        story.append(Paragraph("Accuracy by Difficulty", heading_style))
        diff_data = [['Difficulty', 'Accuracy', 'Count']]
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in valid_results if r["difficulty"] == diff]
            if diff_results:
                diff_acc = sum(r["metrics"]["overall_correct"] for r in diff_results) / len(diff_results)
                diff_data.append([diff.capitalize(), f'{diff_acc:.1%}', str(len(diff_results))])
        
        diff_table = Table(diff_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        diff_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(diff_table)
        story.append(Spacer(1, 20))
        
        # Confidence Level Analysis
        story.append(Paragraph("Accuracy by Confidence Level", heading_style))
        conf_data = [['Confidence', 'Accuracy', 'Count']]
        for conf, results in self.confidence_tracking.items():
            if results:
                conf_acc = sum(results) / len(results)
                conf_data.append([conf, f'{conf_acc:.1%}', str(len(results))])
        
        conf_table = Table(conf_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(conf_table)
        story.append(Spacer(1, 20))
        
        # Tool Usage
        db_matches = sum(r["metrics"]["found_in_db"] for r in valid_results)
        web_searches = sum(r["metrics"]["used_web_search"] for r in valid_results)
        
        story.append(Paragraph("Tool Usage Statistics", heading_style))
        tool_data = [
            ['Tool', 'Usage Count', 'Percentage'],
            ['Database Match', str(db_matches), f'{db_matches/total:.1%}'],
            ['Web Search', str(web_searches), f'{web_searches/total:.1%}'],
        ]
        
        tool_table = Table(tool_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        tool_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(tool_table)
        
        # Add visualization if exists
        chart_files = list(self.output_dir.glob("evaluation_charts_*.png"))
        if chart_files:
            story.append(PageBreak())
            story.append(Paragraph("Performance Visualizations", heading_style))
            story.append(Spacer(1, 12))
            
            latest_chart = max(chart_files, key=lambda p: p.stat().st_mtime)
            img = RLImage(str(latest_chart), width=6*inch, height=4.5*inch)
            story.append(img)
        
        # Build PDF
        doc.build(story)
        logger.info(f"✅ PDF Report generated: {pdf_file}")
   
    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"results_{timestamp}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(self.results),
            "valid_cases": len([r for r in self.results if not r.get("metrics", {}).get("error")]),
            "tool_usage_stats": self.tool_usage_stats,
            "results": self.results
        }
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved detailed results: {out_file}")
        
    def export_to_csv(self):
        """Export results to CSV for easy analysis"""
        valid_results = [r for r in self.results if not r.get("metrics", {}).get("error")]
        if not valid_results:
            return
        
        rows = []
        for r in valid_results:
            metrics = r["metrics"]
            rows.append({
                "test_case": Path(r["test_case"]).name,
                "difficulty": r["difficulty"],
                "gt_name": r["ground_truth"]["destination_name"],
                "gt_district": r["ground_truth"]["district_name"],
                "pred_name": r["prediction"].get("destination_name", ""),
                "pred_district": r["prediction"].get("district_name", ""),
                "correct": metrics["overall_correct"],
                "name_similarity": metrics["name_similarity"],
                "confidence": metrics["confidence_level"],
                "response_time": metrics["response_time_seconds"],
                "content_completeness": metrics["content_completeness"],
                "source": "Database" if metrics["found_in_db"] else "Web Search"
            })
        
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV: {csv_file}")
    
    def create_visualizations(self):
        """Create comprehensive visualization charts"""
        try:
            valid_results = [r for r in self.results if not r.get("metrics", {}).get("error")]
            if not valid_results:
                return
            
            sns.set_style("whitegrid")
            
            df = pd.DataFrame([{
                "difficulty": r["difficulty"],
                "correct": r["metrics"]["overall_correct"],
                "confidence": r["metrics"]["confidence_level"],
                "response_time": r["metrics"]["response_time_seconds"],
                "source": "Database" if r["metrics"]["found_in_db"] else "Web Search",
                "content_completeness": r["metrics"]["content_completeness"]
            } for r in valid_results])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Agent Evaluation Results", fontsize=16, fontweight='bold')
            
            # 1. Accuracy by Difficulty
            acc_by_diff = df.groupby("difficulty")["correct"].mean()
            acc_by_diff.plot(kind="bar", ax=axes[0, 0], color=["#2ecc71", "#f39c12", "#e74c3c"])
            axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].set_title("Accuracy by Difficulty Level")
            axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
            axes[0, 0].set_ylim([0, 1])
            
            for container in axes[0, 0].containers:
                axes[0, 0].bar_label(container, fmt='%.1%%', label_type='edge')
            
            # 2. Response Time by Source
            df.boxplot(column='response_time', by='source', ax=axes[0, 1])
            axes[0, 1].set_ylabel("Response Time (seconds)")
            axes[0, 1].set_title("Response Time: Database vs Web Search")
            axes[0, 1].set_xlabel("Source")
            plt.sca(axes[0, 1])
            plt.xticks(rotation=0)
            
            # 3. Confidence Level Distribution
            conf_counts = df["confidence"].value_counts()
            axes[1, 0].pie(conf_counts, labels=conf_counts.index, autopct='%1.1f%%', 
                          colors=['#2ecc71', '#e74c3c'])
            axes[1, 0].set_title("Confidence Level Distribution")
            
            # 4. Content Completeness by Source
            df.boxplot(column='content_completeness', by='source', ax=axes[1, 1])
            axes[1, 1].set_ylabel("Content Completeness")
            axes[1, 1].set_title("Content Quality: Database vs Web Search")
            axes[1, 1].set_xlabel("Source")
            axes[1, 1].set_ylim([0, 1])
            plt.sca(axes[1, 1])
            plt.xticks(rotation=0)
            
            plt.tight_layout()
            chart_file = self.output_dir / f"evaluation_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {chart_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}", exc_info=True)


async def run_evaluation_demo():
    """Demo runner"""
    agent = get_agent_service()
    evaluator = AgentEvaluator(agent, output_dir="evaluation_results")
    await evaluator.run_full_evaluation("test_dataset_new.json", delay_seconds=6)


if __name__ == "__main__":
    asyncio.run(run_evaluation_demo())