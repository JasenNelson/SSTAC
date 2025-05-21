# Development Instructions for SSTAC Streamlit Web App

## 🚦 **Critical UI/UX and Architectural Guidelines**

To maintain a robust and user-friendly experience, **please adhere to the following rules when making changes or fixing bugs**:

---

### 1. **Sidebar-Only Widgets**
- **All interactive widgets** (file upload, chemical selection, group/media filters, parameter selectors, etc.) **must remain inside a single `with st.sidebar:` block.**
- **Do not move, duplicate, or recreate any widget on the main page.**
- The **main page** is strictly for:
  - Instructions and usage information
  - SSD curve/results display
  - Output summaries and visualizations

---

### 2. **Widget Key Uniqueness**
- **All Streamlit widget keys must be unique** across the app.
- Use a clear key-naming convention (e.g., suffixes like `_file`, `_supabase`) to differentiate widgets for file-upload and database workflows.
- Never use the same key for widgets in different workflows.

---

### 3. **Workflow Separation**
- **File-upload and database workflows must remain logically and visually separated** in the sidebar.
- Do not mix logic or UI elements between these workflows.

---

### 4. **Session State Management**
- Always check and initialize `st.session_state` variables before use.
- Do not remove or bypass session state initialization blocks.

---

### 5. **Error Handling and User Guidance**
- Preserve all user feedback mechanisms (warnings, info, error messages).
- Do not remove or modify user-facing guidance without review.

---

### 6. **Main Page Content Policy**
- **Do not add any widgets to the main page.**
- Main page content should be limited to:
  - Markdown instructions
  - SSD curve plots and results
  - Download buttons for outputs (if needed)
  - No input widgets or selection controls

---

### 7. **Testing After Changes**
- After any modification, thoroughly test both workflows (file-upload and database).
- Confirm that:
  - No DuplicateWidgetID or NameError exceptions occur
  - Widgets appear only in the sidebar
  - The main page remains clean of widgets

---

### 8. **Code Review Checklist**
Before submitting or merging changes:
- [ ] All widgets are inside the sidebar.
- [ ] Widget keys are unique and descriptive.
- [ ] No widgets are present on the main page.
- [ ] Workflow separation is maintained.
- [ ] Error handling and user guidance are intact.
- [ ] All session state variables are properly handled.
- [ ] App runs without UI/UX or runtime errors.

---

## 🛑 **If in Doubt**
If you are unsure whether a change might violate these instructions, **ask the project maintainer before proceeding**.

---

## Example: Sidebar-Only Widgets

```python
with st.sidebar:
    uploaded_file = st.file_uploader(...)
    group_options = st.multiselect(..., key='group_filter_file')
    # ... other widgets ...
# Main page below: only instructions/results
st.markdown("## Instructions")
st.plotly_chart(...)
```

---

**Thank you for helping maintain a clean, robust, and user-friendly application!**
