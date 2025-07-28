import re

def enforce_document_hierarchy(lines_data):
    """
    Corrects the predicted labels based on logical document structure rules,
    including new rules for repeated titles and excessive punctuation.
    """
    if not lines_data:
        return []

    last_active_heading_level = 0
    document_title_text = "" # Store the exact text of the first detected title
    has_document_title = False

    for i, line in enumerate(lines_data):
        current_predicted_label = line['predicted_label']
        line_text = line['text'] # Get the original text for new rules

        # --- New Rule 5: Demote if excessive consecutive punctuation ---
        # Checks for 2 or more consecutive punctuation marks (e.g., "...", "??", "!!")
        if re.search(r'[\.\?,!;:·•-]{2,}', line_text):
            current_predicted_label = 0 # Demote to body text
            line['predicted_label'] = current_predicted_label # Apply immediately to avoid interference
            continue # Move to next line if this rule applies, as it's a strong demotion

        # Rule 1: Demote long text blocks predicted as headings.
        if current_predicted_label > 0 and line['word_count'] > 15:
            current_predicted_label = 0

        # Rule 2: Enforce a single document title (Label 1)
        if current_predicted_label == 1:
            if not has_document_title:
                has_document_title = True
                document_title_text = line_text # Store the original text of the first title
                last_active_heading_level = current_predicted_label
            else:
                # --- New Rule 6: Demote repeated title text ---
                # If a title already exists AND the current line's text exactly matches the title, demote to body.
                # This catches headers/footers with the title.
                if line_text.strip() == document_title_text.strip():
                    current_predicted_label = 0 # Demote to body text
                else:
                    # If it's a label 1 but not the first one, and not exact match, demote to H1
                    current_predicted_label = 2 # Demote to H1
                
        # Rule 3: Enforce logical heading hierarchy for H1, H2, H3 (labels 2, 3, 4)
        elif current_predicted_label > 1: # If it's H1, H2, H3
            # Cannot jump more than one level down (e.g., H1 -> H3 directly is invalid)
            if current_predicted_label > last_active_heading_level + 1:
                current_predicted_label = last_active_heading_level + 1
            
            # If a heading appears after body text, its implied "parent" is the last active heading.
            # If no active heading (last_active_heading_level=0) and it's an H2/H3, promote it to H1.
            if last_active_heading_level == 0 and current_predicted_label > 2:
                 current_predicted_label = 2 # Promote to H1
            
            last_active_heading_level = current_predicted_label

        # Rule 4: If it's body text, reset the last active heading level
        # This is the original rule that caused demotion to H1 after body text.
        # Let's adjust this: only reset `last_active_heading_level` if we expect a brand new top-level section.
        # A common pattern is H1 -> Body -> H2 (this is bad).
        # We want H1 -> Body -> H1 (for a new section).
        # If last_active_heading_level stays at 2 (H1) and an H2 appears, that's fine.
        # So, the aggressive reset to 0 after body text is what makes everything H1.
        # Let's keep `last_active_heading_level` persistent unless explicitly demoted/promoted.
        # The line `if last_active_heading_level == 0 and current_predicted_label > 2:` handles starting new sections.
        # No explicit reset here.

        lines_data[i]['predicted_label'] = current_predicted_label

    return lines_data